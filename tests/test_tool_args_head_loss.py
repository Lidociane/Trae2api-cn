"""Regression: a mid-string argument fragment must not become its own call.

Trae occasionally resumes a native tool call after an upstream hiccup and then
streams only the remaining argument fragment (e.g. ``'/Users/x && ls'`` instead
of ``{"command":"/Users/x && ls"``).  That fragment used to create a second,
argument-less-headed call which aborted the whole turn with a non-retryable
InvalidNativeToolArguments error.
"""

from __future__ import annotations

import unittest

from src import sse


class HeadLossTests(unittest.TestCase):
    def test_mid_string_fragment_is_attached_to_the_open_call(self):
        accumulator = sse.ToolCallAccumulator()
        accumulator.add(
            [{"index": 0, "id": "call_a", "function": {"name": "bash", "arguments": '{"command":"cd /Users/x'}}]
        )
        accumulator.add([{"index": 0, "function": {"name": "bash", "arguments": ' && ls -la"}'}}])
        calls = accumulator.calls()
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0]["function"]["arguments"], '{"command":"cd /Users/x && ls -la"}'
        )

    def test_mid_string_fragment_without_open_call_still_fails_loudly(self):
        accumulator = sse.ToolCallAccumulator()
        accumulator.add([{"index": 0, "function": {"name": "bash", "arguments": "'/Users/x && ls'"}}])
        with self.assertRaises(sse.InvalidNativeToolArguments) as caught:
            sse._ensure_native_tool_arguments(accumulator.calls(), None, strict=True)
        self.assertFalse(caught.exception.retryable)

    def test_unusable_call_does_not_destroy_a_usable_one(self):
        accumulator = sse.ToolCallAccumulator()
        accumulator.add([{"index": 0, "id": "c1", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}])
        accumulator.add([{"index": 0, "id": "c2", "function": {"name": "bash", "arguments": "garbage"}}])
        kept = sse._ensure_native_tool_arguments(accumulator.calls(), None, strict=True)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["function"]["arguments"], '{"command":"ls"}')


if __name__ == "__main__":
    unittest.main()
