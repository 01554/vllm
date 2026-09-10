# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare two tables.py implementations; save compiler artifacts and raw timing.

Gate-closed misses and gate-open hits remain stable across replays. Promotion
cases are correctness-only: repeatedly timing them would turn misses into hits.
This measures the planner, not serving latency or copying expert weights.
"""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def tensors(obj):
    return {
        f.name: getattr(obj, f.name)
        for f in dataclasses.fields(obj)
        if isinstance(getattr(obj, f.name), torch.Tensor)
    }


def state(mod, device, width, gate, experts=128, slots=(4, 4)):
    t = mod.allocate_global_tables(device, experts, slots, width)
    mod.set_gate(t, gate)
    b = mod.allocate_step_buffers(device, experts, width)
    # Detect writes beyond active suffixes, as well as accidental scratch reads.
    for name in ("gather_src", "gather_dst", "staged_expert", "staged_row"):
        getattr(b, name).fill_(-777)
    return t, b


def equal(expected, actual):
    for a, b in zip(expected, actual):
        for name, value in tensors(a).items():
            assert torch.equal(value.cpu(), getattr(b, name).cpu()), name


def correctness(mods, check_weights=False):
    count = 0
    for width in (1, 16, 64):
        cases = [
            [],
            [-1] * width,
            [0] * width,
            list(range(32, 32 + width)),
            [127] * width,
            ([0, 5, 5, -1, 128, -2] * width)[:width],
        ]
        for gate in (False, True):
            for values in cases:
                states = [state(m, "cuda", width, gate) for m in mods]
                ref = state(mods[0], "cpu", width, gate)
                for layer in (0, 1, 0):
                    ids = torch.tensor(values, dtype=torch.int64, device="cuda")
                    weights = (
                        torch.full_like(ids, 0.5, dtype=torch.float32)
                        if check_weights
                        else None
                    )
                    mods[0].step_reference(
                        ref[0],
                        layer,
                        ids.cpu(),
                        ref[1],
                        weights=None if weights is None else weights.cpu(),
                    )
                    for m, (t, b) in zip(mods, states):
                        m.step(t, layer, ids, b, weights=weights)
                    torch.accelerator.synchronize()
                    for actual in states:
                        equal(ref, actual)
                    count += 1
        # Mixed promoted/staged suffix, including nonzero gather offset.
        states = [state(m, "cuda", width, True) for m in mods]
        ref = state(mods[0], "cpu", width, True)
        for m, (t, _) in zip(mods, states):
            m.set_control(t, promote_limit=1)
        mods[0].set_control(ref[0], promote_limit=1)
        values = list(range(32, 32 + width))
        ids = torch.tensor(values, dtype=torch.int64, device="cuda")
        weights = (
            torch.full_like(ids, 0.5, dtype=torch.float32) if check_weights else None
        )
        mods[0].step_reference(
            ref[0],
            1,
            ids.cpu(),
            ref[1],
            weights=None if weights is None else weights.cpu(),
        )
        for m, (t, b) in zip(mods, states):
            m.step(t, 1, ids, b, weights=weights)
            equal(ref, (t, b))
        count += 1
    if check_weights:
        # P1 filters bad weights before distinctness and staging compaction.
        # Compare every table and buffer, including safe_ids and sticky ok/error.
        for gate in (False, True):
            for bad_weight in (float("nan"), float("inf"), -float("inf"), -0.5):
                states = [state(m, "cuda", 16, gate) for m in mods]
                ref = state(mods[0], "cpu", 16, gate)
                for layer in (0, 1, 0):
                    ids = torch.tensor([32, 32, 33, -1, 128, 0], device="cuda")
                    weights = torch.tensor(
                        [bad_weight, 0.5, 0.5, bad_weight, bad_weight, 0.5],
                        device="cuda",
                    )
                    mods[0].step_reference(
                        ref[0], layer, ids.cpu(), ref[1], weights=weights.cpu()
                    )
                    for m, (t, b) in zip(mods, states):
                        m.step(t, layer, ids, b, weights=weights)
                    torch.accelerator.synchronize()
                    for actual in states:
                        equal(ref, actual)
                    count += 1
    return count


class CaptureKernel:
    """Observe the normal launch without changing its arguments or options."""

    def __init__(self, jit):
        self.jit = jit
        self.compiled = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.compiled = self.jit[grid](*args, **kwargs)
            return self.compiled

        return launch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batches", type=int, default=30)
    p.add_argument("--replays", type=int, default=100)
    p.add_argument(
        "--check-weights",
        action="store_true",
        help="Use P1 serving weight validation specialization",
    )
    args = p.parse_args()
    if args.batches < 1 or args.replays < 1:
        p.error("batches and replays must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = (args.baseline, args.candidate)
    mods = [load(path, f"staging_{i}") for i, path in enumerate(paths)]
    manifest = {
        "sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
        "argv": sys.argv,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    count = correctness(mods, args.check_weights)
    (args.output / "correctness.json").write_text(json.dumps({"passed": count}))
    for width in (16, 64):
        for pattern in ("all_hit", "one_staged", "all_staged"):
            states = [
                state(m, "cuda", width, pattern == "all_hit", 512, (258,) * 48)
                for m in mods
            ]
            values = (
                [0] * width
                if pattern == "all_hit"
                else [511] * width
                if pattern == "one_staged"
                else list(range(512 - width, 512))
            )
            ids = torch.tensor(values, dtype=torch.int64, device="cuda")
            weights = (
                torch.full_like(ids, 0.5, dtype=torch.float32)
                if args.check_weights
                else None
            )
            graphs, keepalive = [], []
            for i, (m, (t, b)) in enumerate(zip(mods, states)):
                wrapped = CaptureKernel(m._step_kernel())
                m._KERNELS["step"] = wrapped
                m.step(t, 0, ids, b, weights=weights)
                compiled = wrapped.compiled
                prefix = args.output / f"{width}-{pattern}-{i}"
                for kind, contents in compiled.asm.items():
                    if isinstance(contents, str):
                        prefix.with_suffix(f".{kind}").write_text(contents)
                    elif isinstance(contents, bytes):
                        prefix.with_suffix(f".{kind}").write_bytes(contents)
                prefix.with_suffix(".resources.json").write_text(
                    json.dumps(
                        {
                            "n_regs": compiled.n_regs,
                            "n_spills": compiled.n_spills,
                            "metadata": str(compiled.metadata),
                        },
                        indent=2,
                    )
                )
                m._KERNELS["step"] = wrapped.jit
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    m.step(t, 0, ids, b, weights=weights)
                torch.cuda.current_stream().wait_stream(stream)
                graphs.append(graph)
                keepalive.append((t, b, ids, weights, stream))
            for _ in range(20):
                for g in graphs:
                    g.replay()
            torch.accelerator.synchronize()
            equal(states[0], states[1])
            raw = [[], []]
            for batch in range(args.batches):
                for i in (0, 1) if batch % 2 == 0 else (1, 0):
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(args.replays):
                        graphs[i].replay()
                    end.record()
                    end.synchronize()
                    raw[i].append(start.elapsed_time(end) / args.replays)
            equal(states[0], states[1])
            result = {
                "width": width,
                "pattern": pattern,
                "ms": raw,
                "median_ms": [statistics.median(x) for x in raw],
            }
            with (args.output / "timing.jsonl").open("a") as f:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
