# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the deferred PLE host staging helper."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch


def _load_module() -> Any:
    path = Path(__file__).parents[3] / "vllm/models/qwen4_exp/nvidia/ple_wait.py"
    spec = importlib.util.spec_from_file_location("test_ple_wait_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ple_wait = _load_module()


class _FakeEvent:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def synchronize(self) -> None:
        self.calls.append("event")


class _FakeStream:
    cuda_stream = 1

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
    helper.capacity = int(rows.shape[0])
    helper._active_rows = helper.capacity
    helper._padded_rows = helper.capacity
    return helper


class DeferredRowsTests(unittest.TestCase):
    def test_dummy_signals_before_captured_wait_and_copy(self):
        calls: list[str] = []
        helper = _bare_helper(torch.ones((2, 1, 2)))
        helper._pending = False
        helper._poisoned = False
        helper._gate_armed = False
        helper.flag = torch.zeros(1, dtype=torch.int64)
        helper._ext = _FakeExtension(calls)
        helper._ext.memop_wait_reset = lambda stream, flag: calls.append("wait")
        stream = _FakeStream(calls)
        helper._operation_stream = lambda *args: stream
        helper._validate_destination = lambda destination: None
        destination = type(
            "Destination",
            (),
            {
                "shape": (2, 1, 2),
                "copy_": lambda _, rows, **kwargs: calls.append("copy"),
            },
        )()
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            helper.prepare_dummy(2)
        self.assertEqual(calls, ["stream", "signal"])
        self.assertEqual(helper.rows.count_nonzero().item(), 0)
        helper.consume(destination, capture=True)
        self.assertEqual(calls, ["stream", "signal", "wait", "copy"])

    def test_verify_raw_nan_padding_ids_and_stale_rows(self):
        """Byte diagnostics detect stale rows/IDs without rewriting staging."""
        source = torch.tensor([[[float("nan"), -0.0]]])
        for defect in (None, "row", "ids", "padding"):
            with self.subTest(defect=defect):
                helper = _bare_helper(torch.zeros((2, 1, 2)))
                helper._verify_enabled = True
                helper._verify_ids = torch.tensor([[5]])
                helper.ids = torch.tensor([[5], [0]])
                helper._verify_step = 1
                helper._active_rows = 1
                helper._padded_rows = 2
                helper._pending = False
                helper.flag = torch.zeros(1, dtype=torch.int64)
                helper.destination = torch.cat(
                    (source.clone(), torch.zeros_like(source))
                )
                helper.table = type("Table", (), {"gather": lambda _, ids: source})()
                if defect == "row":
                    helper.destination[0, 0, 1] = 0.0
                elif defect == "ids":
                    helper.ids[0, 0] = 7
                elif defect == "padding":
                    helper.destination[1, 0, 0] = 1.0
                before = helper.destination.view(torch.uint8).clone()
                with (
                    patch.object(
                        torch.cuda, "is_current_stream_capturing", return_value=False
                    ),
                    patch.object(torch.accelerator, "synchronize"),
                ):
                    report = helper.verify_consumed_rows()
                self.assertEqual(report["status"], "pass" if defect is None else "fail")
                self.assertEqual(report["checked_rows"], 1)
                self.assertTrue(
                    torch.equal(before, helper.destination.view(torch.uint8))
                )

    def test_complete_gathers_only_real_candidate_prefix(self) -> None:
        calls: list[str] = []

        class RecordingTable:
            def __init__(self) -> None:
                self.ids: list[int] | None = None

            def gather(self, ids: Any) -> torch.Tensor:
                self.ids = ids.reshape(-1).tolist()
                return torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)

        table = RecordingTable()
        rows = torch.full((8, 2, 2), -1.0)
        helper = _bare_helper(rows)
        helper.table = table
        helper.ids = torch.zeros((8, 2), dtype=torch.int64)
        helper.ids[:3].copy_(torch.tensor([[7, 7], [2, 7], [2, 5]], dtype=torch.int64))
        helper.flag = torch.zeros(1, dtype=torch.int64)
        helper._ext = _FakeExtension(calls)
        helper._readback_event = _FakeEvent(calls)
        helper._pending = True
        helper._rows_ready = False
        helper._active_rows = 3
        helper._padded_rows = 8
        helper._gate_armed = False
        helper._reset_queued = True
        helper._readback_recorded = True
        helper._prepare_stream = None
        helper._poisoned = False
        helper._poison_reason = None

        helper.complete()

        self.assertEqual(table.ids, [7, 7, 2, 7, 2, 5])
        self.assertTrue(
            torch.equal(
                rows[:3], torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
            )
        )
        self.assertTrue(torch.equal(rows[3:], torch.zeros_like(rows[3:])))
        self.assertEqual(calls, ["event", "signal"])

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
