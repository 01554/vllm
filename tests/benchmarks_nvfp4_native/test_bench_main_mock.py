# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive bench.main() on CPU with the GPU pieces stubbed.

Covers the three control paths: a timed shape that passes the replay
check, a shape whose graph replay is corrupted (REPLAY_FAIL, timing
invalidated), and --no-timing. The kernels are not exercised: the runner
returns the oracle's own result (optionally corrupted on replay).
"""

import csv
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import torch

from benchmarks.nvfp4_native import bench
from benchmarks.nvfp4_native.oracle import moe_forward


def make_bank():
    """Tiny raw bank: 3 experts, hidden 32, intermediate 16, per-row fp16 globals."""
    bank = {}
    for prefix, n, k in (("w13", 32, 32), ("w2", 32, 16)):
        bank[f"{prefix}_weight"] = torch.full((3, n, k // 2), 0x22, dtype=torch.uint8)
        bank[f"{prefix}_weight_scale"] = torch.ones((3, n, k // 16)).to(
            torch.float8_e4m3fn
        )
        bank[f"{prefix}_weight_scale_2"] = torch.full(
            (3, n), 0.125, dtype=torch.float16
        )
    bank["w13_weight_scale_2"][:, :16] = 0.25
    return bank


class _Event:
    def __init__(self, enable_timing=False):
        self.t = 0.0

    def record(self):
        self.t = _Clock.tick()

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return other.t - self.t


class _Clock:
    now = 0.0

    @classmethod
    def tick(cls):
        cls.now += 1.0
        return cls.now


class _Stream:
    def __init__(self, device=None):
        pass

    def wait_stream(self, other):
        pass


class _Graph:
    """Records the runner call inside capture and re-runs it on replay."""

    def __init__(self):
        self.fn = None

    def replay(self):
        assert self.fn is not None
        self.fn()


@contextmanager
def _graph_ctx(graph, stream=None):
    holder: dict[str, Any] = {}
    _CaptureState.current = holder
    try:
        yield
    finally:
        _CaptureState.current = None
        graph.fn = holder["fn"]


class _CaptureState:
    current: dict[str, Any] | None = None


class FakeRunner:
    """Returns the oracle result; `corrupt_replays` flips the replayed output."""

    name = "fake"

    def __init__(self, bank, corrupt_replays=False):
        self.bank = bank
        self.device = torch.device("cpu")
        self.error = torch.zeros(1, dtype=torch.int32)
        self.corrupt_replays = corrupt_replays
        self._out: torch.Tensor | None = None
        self._calls = 0

    def path(self, m):
        return "gemv" if m == 1 else "grouped"

    def prepare(self, m):
        return None

    def __call__(self, x, ids, w):
        b = self.bank
        result = moe_forward(
            x,
            ids,
            w,
            b["w13_weight"],
            b["w13_weight_scale"],
            b["w13_weight_scale_2"],
            b["w2_weight"],
            b["w2_weight_scale"],
            b["w2_weight_scale_2"],
        )
        if _CaptureState.current is not None:
            # capture: keep an output tensor that replays overwrite in place
            self._out = result.clone()

            def replay():
                self._calls += 1
                assert self._out is not None
                if self.corrupt_replays:
                    self._out.add_(1.0)
                else:
                    self._out.copy_(result)

            _CaptureState.current["fn"] = replay
            return self._out
        return result


class BenchMainMockTests(unittest.TestCase):
    def run_main(
        self, tmp, corrupt=False, extra=(), runner_factory=None, backends="native"
    ):
        bank = make_bank()  # 3 experts, hidden 32, intermediate 16

        def fake_load(shard, prefix, num_experts, experts=None):
            return bank, {
                "shard": shard,
                "prefix": prefix,
                "experts": list(range(num_experts)),
            }

        def make_runner(name):
            if runner_factory is not None:
                return runner_factory(bank)
            return FakeRunner(bank, corrupt_replays=corrupt)

        argv = [
            "bench",
            "--shard",
            "x.safetensors",
            "--prefix",
            "p",
            "--num-experts",
            "3",
            "--out",
            tmp,
            "--backends",
            backends,
            "--sizes",
            "1,3",
            "--device",
            "cpu",
            *(
                ()
                if any(e.startswith("--patterns") for e in extra)
                else ("--patterns", "uniform")
            ),
            *extra,
        ]
        with (
            mock.patch.object(bench, "load_layer_bank", fake_load),
            mock.patch.object(
                bench,
                "NativeRunner",
                lambda bank, device, gemv_rows=1: make_runner("native"),
            ),
            mock.patch.object(
                bench,
                "KernelRunner",
                lambda bank, device, gemv_rows=1: make_runner("native_kernel"),
            ),
            mock.patch.object(bench, "TOP_K", 4),
            mock.patch.object(bench, "WARMUP", 2),
            mock.patch.object(bench, "BATCHES", 3),
            mock.patch.object(bench, "REPLAYS_PER_BATCH", 2),
            mock.patch.object(torch.cuda, "Stream", _Stream),
            mock.patch.object(torch.cuda, "CUDAGraph", _Graph),
            mock.patch.object(torch.cuda, "Event", _Event),
            mock.patch.object(torch.cuda, "graph", _graph_ctx),
            mock.patch.object(torch.cuda, "stream", lambda s: _graph_ctx(_Graph())),
            mock.patch.object(torch.cuda, "current_stream", lambda d=None: _Stream()),
            mock.patch.object(torch.accelerator, "synchronize", lambda d=None: None),
            mock.patch.object(bench, "make_inputs", self.make_inputs),
            mock.patch.object(sys, "argv", argv),
        ):
            bench.main()
        with open(Path(tmp) / "summary.csv") as fh:
            rows = list(csv.DictReader(fh))
        correctness = json.loads((Path(tmp) / "correctness.json").read_text())
        return rows, correctness

    @staticmethod
    def make_inputs(m, hidden, num_experts, seed, pattern="uniform", **_kw):
        g = torch.Generator().manual_seed(seed)
        x = (torch.randn((m, hidden), generator=g) * 0.5).to(torch.bfloat16)
        pool = torch.arange(num_experts) if pattern == "uniform" else torch.arange(3)
        ids = torch.stack(
            [pool[torch.randperm(len(pool), generator=g)[:3]] for _ in range(m)]
        )
        ids = torch.cat((ids, torch.full((m, 1), -1)), dim=1).to(
            torch.int32
        )  # padding route
        w = torch.softmax(torch.rand((m, 4), generator=g), dim=-1)
        return x, ids, w

    def test_timed_run_passes_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, correctness = self.run_main(tmp)
            self.assertEqual([r["correct"] for r in rows], ["PASS", "PASS"])
            self.assertEqual([r["source_equivalent"] for r in rows], ["yes", "yes"])
            self.assertTrue(all(r["median_ms"] for r in rows))
            self.assertTrue(all(correctness[k]["pass"] for k in correctness))
            for f in (
                "manifest.json",
                "timing.jsonl",
                "command.txt",
                "inputs/uniform_m1.pt",
            ):
                self.assertTrue((Path(tmp) / f).exists(), f)
            timing = [
                json.loads(line)
                for line in (Path(tmp) / "timing.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(t["valid"] for t in timing))

    def test_corrupted_replay_invalidates_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = self.run_main(tmp, corrupt=True)
            self.assertEqual(
                [r["correct"] for r in rows], ["REPLAY_FAIL", "REPLAY_FAIL"]
            )
            self.assertEqual([r["median_ms"] for r in rows], ["", ""])
            timing = [
                json.loads(line)
                for line in (Path(tmp) / "timing.jsonl").read_text().splitlines()
            ]
            self.assertFalse(any(t["valid"] for t in timing))
            self.assertIsNone(timing[0]["median_ms"])

    def test_non_equivalent_backend_is_marked_in_summary_and_timing(self):
        class NonEquivalentRunner(FakeRunner):
            globals_report = {"source_equivalent": False}

        with tempfile.TemporaryDirectory() as tmp:
            rows, correctness = self.run_main(
                tmp, runner_factory=lambda bank: NonEquivalentRunner(bank)
            )
            self.assertEqual([r["source_equivalent"] for r in rows], ["NO", "NO"])
            self.assertEqual([r["correct"] for r in rows], ["PASS", "PASS"])
            self.assertTrue(
                all(c["source_equivalent"] is False for c in correctness.values())
            )
            timing = [
                json.loads(line)
                for line in (Path(tmp) / "timing.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(t["source_equivalent"] is False for t in timing))

    def test_two_patterns_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, correctness = self.run_main(
                tmp, extra=("--patterns", "uniform,working_set")
            )
            self.assertEqual(
                [(r["pattern"], r["M"]) for r in rows],
                [
                    ("uniform", "1"),
                    ("uniform", "3"),
                    ("working_set", "1"),
                    ("working_set", "3"),
                ],
            )
            self.assertEqual(
                sorted(correctness),
                sorted(
                    f"native/{p}/M{m}"
                    for p in ("uniform", "working_set")
                    for m in (1, 3)
                ),
            )
            self.assertTrue((Path(tmp) / "inputs" / "working_set_m3.pt").exists())
            timing = [
                json.loads(line)
                for line in (Path(tmp) / "timing.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                sorted({t["pattern"] for t in timing}), ["uniform", "working_set"]
            )

    def test_two_backends_are_timed_with_alternating_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = self.run_main(tmp, backends="native,native_kernel")
            self.assertEqual(
                [(r["backend"], r["M"]) for r in rows],
                [
                    ("native", "1"),
                    ("native_kernel", "1"),
                    ("native", "3"),
                    ("native_kernel", "3"),
                ],
            )
            timing = [
                json.loads(line)
                for line in (Path(tmp) / "timing.jsonl").read_text().splitlines()
            ]
            for t in timing:
                self.assertEqual(
                    t["interleaved_with"],
                    [b for b in ("native", "native_kernel") if b != t["backend"]],
                )
                orders = t["batch_order"]
                self.assertEqual(len(orders), 3)  # BATCHES patched to 3
                self.assertEqual(orders[0], ["native", "native_kernel"])
                self.assertEqual(orders[1], ["native_kernel", "native"])
                self.assertEqual(orders[2], ["native", "native_kernel"])
                self.assertEqual(len(t["samples_ms"]), 3)

    def test_no_timing_only_writes_correctness(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, correctness = self.run_main(tmp, extra=("--no-timing",))
            self.assertEqual([r["correct"] for r in rows], ["PASS", "PASS"])
            self.assertEqual(len(correctness), 2)
            self.assertEqual((Path(tmp) / "timing.jsonl").read_text(), "")
