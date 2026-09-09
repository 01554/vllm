# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the deferred PLE host staging helper."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import Mock, patch

import torch

from vllm.models.qwen4_exp.nvidia import ple_wait


class _FakeEvent:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def synchronize(self) -> None:
        self.calls.append("event")


class _FakeStream:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def synchronize(self) -> None:
        self.calls.append("stream")


class _FakeExtension:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def signal_flag(self, _flag_ptr: int) -> None:
        self.calls.append("signal")


def _bare_helper(rows: torch.Tensor) -> Any:
    helper = object.__new__(ple_wait.DeferredRows)
    helper.rows = rows
    return helper


class DeferredRowsTests(unittest.TestCase):
    def test_probe_unsupported_wait_fences_queued_write(self) -> None:
        helper = _bare_helper(torch.empty(1))
        helper._ext = Mock()
        helper._ext.memop_write.return_value = 0
        helper._ext.memop_wait_geq.return_value = 801
        stream = Mock(cuda_stream=12)
        scratch = torch.zeros(1, dtype=torch.int64)
        with (
            patch.object(ple_wait.torch, "zeros", return_value=scratch),
            self.assertRaises(ple_wait.StreamMemopsUnavailable),
        ):
            helper._probe_stream_memops(stream)
        stream.synchronize.assert_called_once_with()

    def test_probe_driver_error_is_not_capability_fallback(self) -> None:
        helper = _bare_helper(torch.empty(1))
        helper._ext = Mock()
        helper._ext.memop_write.return_value = 1
        stream = Mock(cuda_stream=12)
        scratch = torch.zeros(1, dtype=torch.int64)
        with (
            patch.object(ple_wait.torch, "zeros", return_value=scratch),
            self.assertRaises(RuntimeError) as caught,
        ):
            helper._probe_stream_memops(stream)
        self.assertNotIsInstance(caught.exception, ple_wait.StreamMemopsUnavailable)
        helper._ext.memop_wait_geq.assert_not_called()
        stream.synchronize.assert_called_once_with()

    def test_raw_uint8_rows_reinterpret_for_bfloat16_and_fp8(self) -> None:
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            rows = torch.empty((1, 2, 3), dtype=dtype)
            raw = torch.arange(
                rows.numel() * rows.element_size(), dtype=torch.uint8
            ).reshape(2, -1)
            actual = _bare_helper(rows)._as_rows_tensor(raw)
            expected = raw.reshape(-1).view(dtype).reshape(rows.shape)
            self.assertTrue(
                torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            )

    def test_raw_uint8_rows_validate_byte_count(self) -> None:
        rows = torch.empty((1, 2, 3), dtype=torch.bfloat16)
        helper = _bare_helper(rows)
        with self.assertRaisesRegex(ValueError, "bytes"):
            helper._as_rows_tensor(torch.zeros((2, 5), dtype=torch.uint8))

    def test_abort_fences_queued_reset_before_signal(self) -> None:
        calls: list[str] = []
        helper = _bare_helper(torch.zeros((1, 1, 2), dtype=torch.bfloat16))
        helper._ext = _FakeExtension(calls)
        helper.flag = torch.zeros(1, dtype=torch.int64)
        helper._readback_event = _FakeEvent(calls)
        helper._prepare_stream = _FakeStream(calls)
        helper._pending = True
        helper._rows_ready = False
        helper._gate_armed = False
        helper._reset_queued = True
        helper._readback_recorded = True
        helper._poisoned = False
        helper._poison_reason = None

        helper.abort()

        self.assertEqual(calls, ["event", "signal"])
        self.assertTrue(helper.poisoned)
        self.assertFalse(helper.pending)
        self.assertFalse(helper._reset_queued)

    def test_abort_fences_prepare_partial_failure_before_signal(self) -> None:
        calls: list[str] = []
        helper = _bare_helper(torch.zeros((1, 1, 2), dtype=torch.bfloat16))
        helper._ext = _FakeExtension(calls)
        helper.flag = torch.zeros(1, dtype=torch.int64)
        helper._readback_event = _FakeEvent(calls)
        helper._prepare_stream = _FakeStream(calls)
        helper._pending = False
        helper._rows_ready = True
        helper._gate_armed = False
        helper._reset_queued = True
        helper._readback_recorded = False
        helper._poisoned = False
        helper._poison_reason = None

        helper.abort()

        self.assertEqual(calls, ["stream", "signal"])
        self.assertTrue(helper.poisoned)


if __name__ == "__main__":
    unittest.main()
