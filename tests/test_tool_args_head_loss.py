"""Regression: a mid-string argument fragment must not become its own call.

Trae occasionally resumes a native tool call after an upstream hiccup (gateway
fallback / plan-item switch) and then streams only the remaining argument
fragment, e.g. ``/Users/x && ls"}`` instead of ``{"command":"/Users/x && ls"}``.

Before the fix the accumulator turned that fragment into a second, head-less
call and ``_ensure_native_tool_arguments(strict=True)`` aborted the whole turn
with a non-retryable ``InvalidNativeToolArguments`` error.
"""

from __future__ import annotations

import json
import unittest

from src import sse
from tests.test_ide_native_tool_deltas import TOOLS, _call, _response


def _framed(*calls: dict) -> list[list[dict]]:
    return [[call] for call in calls]


async def _stream_calls(frames: list[list[dict]], stream: bool) -> list[dict]:
    if not stream:
        result = await sse.collect_nonstream_ide(
            _response(frames), "m", allowed_tools=TOOLS, fail_on_empty=True
        )
        return result["choices"][0]["message"]["tool_calls"] or []
    merged: dict[int, dict] = {}
    async for chunk in sse.translate_ide_stream(
        _response(frames), "m", allowed_tools=TOOLS, fail_on_empty=True
    ):
        if not chunk.startswith("data: "):
            continue
        body = chunk[6:].strip()
        if not body or body == "[DONE]":
            continue
        payload = json.loads(body)
        for choice in payload["choices"]:
            for delta in choice["delta"].get("tool_calls", []):
                slot = merged.setdefault(
                    delta["index"],
                    {"id": delta["id"], "type": "function",
                     "function": {"name": "", "arguments": ""}},
                )
                function = delta.get("function") or {}
                for key in ("name", "arguments"):
                    slot["function"][key] += function.get(key) or ""
    return list(merged.values())


class HeadLossAccumulatorTests(unittest.TestCase):
    def test_mid_string_fragment_is_attached_to_the_open_call(self):
        accumulator = sse.ToolCallAccumulator()
        accumulator.add([{"index": 0, "id": "call_a",
                          "function": {"name": "bash", "arguments": '{"command":"cd /Users/x'}}])
        accumulator.add([{"index": 0, "function": {"name": "bash", "arguments": ' && ls -la"}'}}])
        calls = accumulator.calls()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["arguments"], '{"command":"cd /Users/x && ls -la"}')

    def test_plan_item_switch_mid_call_is_healed(self):
        """A resumed fragment carrying a different plan item used to split the call."""
        accumulator = sse.ToolCallAccumulator()
        accumulator.add([{"index": 0, "id": "call_a",
                          "function": {"name": "read_file", "arguments": '{"path":"/Users/x'},
                          "_event_session_id": "s1", "_plan_item_id": "p1"}])
        accumulator.add([{"index": 0,
                          "function": {"name": "read_file", "arguments": '/y.md"}'},
                          "_event_session_id": "s2", "_plan_item_id": "p2"}])
        calls = accumulator.calls()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["arguments"], '{"path":"/Users/x/y.md"}')

    def test_unusable_call_does_not_destroy_a_usable_one(self):
        accumulator = sse.ToolCallAccumulator()
        accumulator.add([{"index": 0, "id": "c1",
                          "function": {"name": "bash", "arguments": '{"command":"ls"}'}}])
        accumulator.add([{"index": 0, "id": "c2",
                          "function": {"name": "bash", "arguments": "garbage"}}])
        kept = sse._ensure_native_tool_arguments(accumulator.calls(), None, strict=True)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["function"]["arguments"], '{"command":"ls"}')


class HeadLossAmbiguityTests(unittest.TestCase):
    """Ambiguous attachments must be refused, never guessed."""

    def test_fragment_with_two_open_candidates_is_not_attached(self):
        accumulator = sse.ToolCallAccumulator()
        accumulator.add([{"index": 0, "id": "call_a",
                          "function": {"name": "bash", "arguments": '{"command":"cd /Users/x'}}])
        accumulator.add([{"index": 1, "id": "call_b",
                          "function": {"name": "bash", "arguments": '{"command":"echo hi'}}])
        # 无 index 的缺头续片：两个调用都可能，必须放弃接回而不是猜
        accumulator.add([{"function": {"name": "bash", "arguments": ' && tail"'}}])
        kept = sse._ensure_native_tool_arguments(accumulator.calls(), None, strict=True)
        self.assertEqual(len(kept), 2)
        for call in kept:
            self.assertEqual(json.loads(call["function"]["arguments"]).__class__, dict)

    def test_heredoc_payload_is_recovered_after_plan_switch(self):
        command = "python3 - <<'PY'\nimport json\nd = {'a': \"x y\", 'b': [1, 2]}\nprint(json.dumps(d, ensure_ascii=False))\nPY"
        payload = json.dumps({"command": command}, ensure_ascii=False)
        head, tail = payload[: len(payload) // 2], payload[len(payload) // 2 :]
        accumulator = sse.ToolCallAccumulator()
        accumulator.add([{"index": 0, "id": "call_h",
                          "function": {"name": "bash", "arguments": head},
                          "_event_session_id": "s1", "_plan_item_id": "p1"}])
        accumulator.add([{"index": 0,
                          "function": {"name": "bash", "arguments": tail},
                          "_event_session_id": "s2", "_plan_item_id": "p2"}])
        kept = sse._ensure_native_tool_arguments(accumulator.calls(), None, strict=True)
        self.assertEqual(len(kept), 1)
        self.assertEqual(json.loads(kept[0]["function"]["arguments"])["command"], command)


class HeadLossStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_resumed_fragment_completes_the_turn(self):
        frames = _framed(
            _call('{"path":"/Users/x', name="read_file", call_id="call_native"),
            _call('/y.md"}', name="read_file"),
        )
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await _stream_calls(frames, stream)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["function"]["arguments"], '{"path":"/Users/x/y.md"}')

    async def test_orphan_tail_still_fails_loudly(self):
        frames = _framed(_call('/y.md"}', name="read_file", call_id="call_native"))
        for stream in (False, True):
            with self.subTest(stream=stream):
                with self.assertRaises(sse.InvalidNativeToolArguments) as caught:
                    await _stream_calls(frames, stream)
                self.assertFalse(caught.exception.retryable)

    async def test_a_broken_call_no_longer_kills_the_whole_turn(self):
        frames = _framed(
            _call('{"path":"a.md"}', name="read_file", call_id="call_good"),
            _call("garbage-without-head", name="read_file", call_id="call_bad"),
        )
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await _stream_calls(frames, stream)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["function"]["arguments"], '{"path":"a.md"}')

    async def test_long_command_split_at_many_offsets_is_recovered(self):
        command = "cd /Users/gezhijie/.dsh && " + " ; ".join(
            f"echo 'step {i} with \"quotes\" and spaces'" for i in range(60)
        )
        payload = json.dumps({"command": command}, ensure_ascii=False)
        self.assertGreater(len(payload), 2000)
        head, tail = payload[: len(payload) // 2], payload[len(payload) // 2:]
        frames = _framed(
            _call(head, name="read_file", call_id="call_long"),
            _call(tail, name="read_file"),
        )
        for stream in (False, True):
            with self.subTest(stream=stream):
                calls = await _stream_calls(frames, stream)
                self.assertEqual(len(calls), 1)
                self.assertEqual(json.loads(calls[0]["function"]["arguments"])["command"], command)


if __name__ == "__main__":
    unittest.main()
