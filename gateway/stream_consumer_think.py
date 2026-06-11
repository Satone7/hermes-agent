"""Think-block filtering for GatewayStreamConsumer.

Some models emit inline <think>...</think> blocks in content.  The agent strips
them from the final response, but intermediate edits go out before that, so this
mirrors the CLI's _stream_delta state machine; tag primitives are shared with
``agent/think_scrubber.py`` so the progressive display matches the post-stream scrubber."""

from __future__ import annotations

import logging

from agent.think_scrubber import (
    GEMMA_CHANNEL_OPEN, THINK_CLOSE_TAGS, THINK_OPEN_TAGS, normalize_gemma_channel_tokens,
)

from agent.think_scrubber import StreamingThinkScrubber as _Scrubber

logger = logging.getLogger("gateway.stream_consumer")


class StreamThinkFilterMixin:
    """Progressive <think>-tag suppression over streamed deltas."""

    _OPEN_THINK_TAGS = THINK_OPEN_TAGS
    _CLOSE_THINK_TAGS = THINK_CLOSE_TAGS
    # Hold-back candidates for a partial opener at the buffer tail.  The Gemma control token
    # normalizes to <think> only once complete, so its fragments must be held too — otherwise a
    # deltas-split token emits its leading "<|chan" as visible text before the rewrite can fire.
    _PARTIAL_OPEN_TOKENS = THINK_OPEN_TAGS + (GEMMA_CHANNEL_OPEN,)

    def _at_block_boundary(self, buf: str, idx: int) -> bool:
        """Tag at ``idx`` starts a block: start of text, or newline + optional whitespace.

        Prose that merely *mentions* a tag must not trigger.
        """
        acc_boundary = not self._accumulated or self._accumulated.endswith("\n")
        if idx == 0:
            return acc_boundary
        preceding = buf[:idx]
        last_nl = preceding.rfind("\n")
        if last_nl == -1:
            return acc_boundary and preceding.strip() == ""
        return preceding[last_nl + 1:].strip() == ""

    def _earliest_open_tag(self, buf: str, lower_buf: str) -> "tuple[int, int]":
        """(index, length) of the earliest block-boundary opening tag, or (-1, 0)."""
        best_idx, best_len = -1, 0
        for tag in self._OPEN_THINK_TAGS:
            tag_lower = tag.lower()
            search_start = 0
            while (idx := lower_buf.find(tag_lower, search_start)) != -1:
                if self._at_block_boundary(buf, idx):
                    if best_idx == -1 or idx < best_idx:
                        best_idx, best_len = idx, len(tag)
                    break  # first boundary hit for this tag is enough
                search_start = idx + 1
        return best_idx, best_len

    def _filter_and_accumulate(self, text: str) -> None:
        """Append a delta to the buffer, discarding think blocks.

        Partial tags at buffer boundaries are held in ``_think_buffer`` until
        enough characters arrive to decide.
        """
        # Gemma 4 marks reasoning with control tokens rather than angle-bracket tags; rewrite
        # them to <think>/</think> here so the shared state machine handles them transparently.
        # A token split across deltas stays in _think_buffer (see _PARTIAL_TAGS) until resolved.
        buf = normalize_gemma_channel_tokens(self._think_buffer + text)
        self._think_buffer = ""

        while buf:
            # Case-insensitive: models emit <Think>, <THINKING>, …
            lower_buf = buf.lower()
            if self._in_think_block:
                best_idx, best_len = _Scrubber._find_first_tag(buf, self._CLOSE_THINK_TAGS)
                if best_len:
                    self._in_think_block = False
                    buf = buf[best_idx + best_len:]
                else:
                    # Hold a tail that could be a partial close tag; discard the rest. A RAW
                    # Gemma close fragment (`<chan`) is covered by the same window: normalization
                    # runs on the reassembled buffer, and the token is no longer than </think>.
                    max_tag = max(len(t) for t in self._CLOSE_THINK_TAGS)
                    self._think_buffer = buf[-max_tag:] if len(buf) > max_tag else buf
                    return
            else:
                best_idx, best_len = self._earliest_open_tag(buf, lower_buf)
                if best_len:
                    self._append_accumulated(buf[:best_idx])
                    self._in_think_block = True
                    buf = buf[best_idx + best_len:]
                else:
                    # Hold back a partial open tag at the tail.
                    held_back = _Scrubber._max_partial_suffix(buf, self._PARTIAL_OPEN_TOKENS)
                    if held_back:
                        self._append_accumulated(buf[:-held_back])
                        self._think_buffer = buf[-held_back:]
                    else:
                        # An orphan </think> (thinking-mode toggle dropped the open, or
                        # incomplete upstream stripping) is noise.
                        self._append_accumulated(self._strip_orphan_close_tags(buf))
                    return

    @staticmethod
    def _strip_orphan_close_tags(text: str) -> str:
        """Remove close tags (plus trailing whitespace) that have no matching open."""
        return _Scrubber._strip_orphan_close_tags(text)

    def _flush_think_buffer(self) -> None:
        """On stream end, flush text held back waiting for a possible open tag."""
        if self._think_buffer and not self._in_think_block:
            self._append_accumulated(self._strip_orphan_close_tags(self._think_buffer))
            self._think_buffer = ""
