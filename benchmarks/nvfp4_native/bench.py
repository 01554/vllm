# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident one-layer NVFP4 MoE microbenchmark (native kernels vs Marlin).

Protocol: notes/upstream-native-kernel-benchmark-recipe-2026-09-09.md.
Correctness against the independent CPU oracle gates every timed shape;
a failure stops the run for that backend (no tolerance changes here).
Timing replays the complete MoE call inside a CUDA graph: warmup, then
30 batches of 20 replays with CUDA events around each batch. Reported
values are batch-mean per-call latencies (median, p10, p90, variance).

Usage (GPU host):
  python -m benchmarks.nvfp4_native.bench --shard <model-00001-of-00010.safetensors> \
      --prefix model.language_model.layers.0.mlp --num-experts 512 --out <dir> \
      [--backends native,native_apply,marlin] [--sizes 1,2,4,8,16,64,256,1024]

Backends: "native" calls the kernels with independently allocated
workspaces (package-level pre-check); "native_apply" goes through
NativeNvFp4Experts.apply with a caller-provided scratch (the experts
path without the worker's WorkspaceManager); "marlin" is the reference
backend. Correctness is reported against the source-semantics oracle
(float32 globals) and, as a second column, against an oracle that uses the
backend's float16 per-row globals; the pass verdict uses the source column.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

from benchmarks.nvfp4_native.layer_loader import load_layer_bank, write_manifest
from benchmarks.nvfp4_native.oracle import moe_forward as oracle_forward

TOP_K = 10
WARMUP = 20
BATCHES = 30
REPLAYS_PER_BATCH = 20


def sha256_tensor(t: torch.Tensor) -> str:
    # reshape(-1) first: 0-dim scalars (the F32 global/input scales) cannot
    # be viewed as bytes directly.
    flat = t.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def make_inputs(m: int, hidden: int, num_experts: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn((m, hidden), generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(num_experts, generator=g)[:TOP_K] for _ in range(m)]
    ).to(torch.int32)  # distinct valid routes per token
    w = torch.softmax(torch.rand((m, TOP_K), generator=g), dim=-1).to(torch.float32)
    return x, ids, w


class NativeRunner:
    """Native decode GEMV (M <= gemv_rows) / grouped prefill on the raw bank."""

    name = "native"

    def __init__(self, bank, device, gemv_rows: int = 1):
        from vllm.model_executor.layers.quantization.nvfp4_native import bank as nb
        from vllm.model_executor.layers.quantization.nvfp4_native import prefill as npf
        from vllm.model_executor.layers.quantization.nvfp4_native.loader import (
            expand_w2_globals,
            expand_w13_globals,
        )

        self.nb, self.npf = nb, npf
        rows, n2, kh = bank["w13_weight"].shape
        intermediate, hidden = n2 // 2, kh * 2
        # The backend's per-row float16 globals (recorded as a transformation).
        self.bank = {
            "w13_weight": bank["w13_weight"].to(device),
            "w2_weight": bank["w2_weight"].to(device),
            "w13_weight_scale": bank["w13_weight_scale"].to(device),
            "w2_weight_scale": bank["w2_weight_scale"].to(device),
            "w13_weight_scale_2": expand_w13_globals(
                bank["w13_weight_scale_2"], intermediate
            ).to(device),
            "w2_weight_scale_2": expand_w2_globals(
                bank["w2_weight_scale_2"], hidden
            ).to(device),
        }
        self.rows = rows
        self.gemv_rows = gemv_rows
        self.device = device
        self.step_map = torch.arange(rows, dtype=torch.int32, device=device)
        self.error = torch.zeros(1, dtype=torch.int32, device=device)
        self._ws = {}

    def path(self, m: int) -> str:
        return "gemv" if m <= self.gemv_rows else "grouped"

    def prepare(self, m: int):
        if m not in self._ws:
            if m <= self.gemv_rows:
                self._ws[m] = self.nb.allocate_workspace(
                    self.bank, m, TOP_K, num_experts=self.rows
                )
            else:
                self._ws[m] = self.npf.allocate_workspace(
                    self.bank, m, TOP_K, num_experts=self.rows
                )
        return self._ws[m]

    def __call__(self, x, ids, w):
        ws = self.prepare(x.shape[0])
        if x.shape[0] <= self.gemv_rows:
            return self.nb.gemv(x, w, ids, self.bank, self.step_map, ws)
        return self.npf.prefill(x, w, ids, self.bank, self.step_map, ws)


class ExpertsApplyRunner(NativeRunner):
    """NativeNvFp4Experts.apply() with a caller-provided scratch buffer.

    Mirrors the modular-kernel path (workspace_shapes -> carve in apply)
    without the worker's WorkspaceManager; integration timing through the
    full runner is the model-launch validation, not this bench.
    """

    name = "native_apply"

    def __init__(self, bank, device, gemv_rows: int = 1):
        super().__init__(bank, device, gemv_rows)
        from types import SimpleNamespace

        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.quantization.nvfp4_native.experts import (
            NativeNvFp4Experts,
        )

        self._act = MoEActivation.SILU
        self.experts = NativeNvFp4Experts.__new__(NativeNvFp4Experts)
        self.experts.moe_config = SimpleNamespace(
            experts_per_token=TOP_K, max_num_tokens=1024
        )
        self.experts.quant_config = SimpleNamespace(
            gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None
        )
        self.experts.gemv_rows = gemv_rows
        self.experts._bank = None
        self.experts._step_map = None
        self.experts._error = None
        self.experts._rows = self.experts._hidden = self.experts._intermediate = 0
        layer = SimpleNamespace(
            **{k: SimpleNamespace(data=v) for k, v in self.bank.items()}
        )
        self.experts.process_weights_after_loading(layer)
        self.error = self.experts._error
        self._scratch = {}
        self._out = {}

    def prepare(self, m: int):
        if m not in self._scratch:
            hidden = self.bank["w13_weight"].shape[2] * 2
            _w1, (elems,), _o = self.experts.workspace_shapes(
                m, hidden // 2, hidden, TOP_K, self.rows, self.rows, None, self._act
            )
            self._scratch[m] = torch.empty(
                (elems,), dtype=torch.bfloat16, device=self.device
            )
            self._out[m] = torch.empty(
                (m, hidden), dtype=torch.bfloat16, device=self.device
            )
        return self._scratch[m]

    def __call__(self, x, ids, w):
        m = x.shape[0]
        scratch = self.prepare(m)
        out = self._out[m]
        self.experts.apply(
            out,
            x,
            self.bank["w13_weight"],
            self.bank["w2_weight"],
            w,
            ids,
            self._act,
            self.rows,
            None,
            None,
            None,
            torch.empty(0, device=self.device),
            scratch,
            None,
            False,
        )
        return out


class MarlinRunner:
    """Marlin NVFP4 MoE on the same bank (repacked); present only if importable."""

    name = "marlin"

    def __init__(self, bank, device):
        raise NotImplementedError(
            "Marlin path pending: needs the Marlin repack + fused_marlin_moe call "
            "wired for this bank (tracked in the recipe note)."
        )


def _oracle(bank, x, ids, w, f16_globals: bool):
    g13, g2 = bank["w13_weight_scale_2"], bank["w2_weight_scale_2"]
    if f16_globals:
        # The backend's per-row float16 globals (the loader's transformation).
        from vllm.model_executor.layers.quantization.nvfp4_native.loader import (
            expand_w2_globals,
            expand_w13_globals,
        )

        intermediate = bank["w13_weight"].shape[1] // 2
        hidden = bank["w13_weight"].shape[2] * 2
        g13 = expand_w13_globals(g13, intermediate)
        g2 = expand_w2_globals(g2, hidden)
    return oracle_forward(
        x,
        ids,
        w,
        bank["w13_weight"],
        bank["w13_weight_scale"],
        g13,
        bank["w2_weight"],
        bank["w2_weight_scale"],
        g2,
    )


def _compare(got: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> dict:
    diff = (got - ref).abs()
    tol = atol + rtol * ref.abs()
    return {
        "max_abs_diff": float(diff.max()),
        "max_rel_diff": float((diff / (ref.abs() + 1e-6)).max()),
        "normalized_rms": float(
            diff.pow(2).mean().sqrt() / (ref.pow(2).mean().sqrt() + 1e-12)
        ),
        "violations": int((diff > tol).sum()),
        "elements": int(diff.numel()),
        "pass": bool((diff <= tol).all()),
    }


def check_correctness(runner, bank, m, x, ids, w, atol, rtol):
    if getattr(runner, "error", None) is not None:
        runner.error.zero_()
    out = runner(x.to(runner.device), ids.to(runner.device), w.to(runner.device))
    torch.accelerator.synchronize(runner.device)
    got = out.detach().cpu().float()
    source = _compare(got, _oracle(bank, x, ids, w, False).float(), atol, rtol)
    backend_globals = _compare(got, _oracle(bank, x, ids, w, True).float(), atol, rtol)
    sticky = (
        int(runner.error.item()) if getattr(runner, "error", None) is not None else None
    )
    return {
        "vs_source_f32_globals": source,
        "vs_backend_f16_globals": backend_globals,
        "sticky_error": sticky,
        "non_finite": int((~torch.isfinite(got)).sum()),
        "pass": source["pass"]
        and (sticky in (None, 0))
        and bool(torch.isfinite(got).all()),
        "output_sha256": sha256_tensor(out),
    }


def time_graph(runner, m, x, ids, w):
    device = runner.device
    xd, idd, wd = x.to(device), ids.to(device), w.to(device)
    runner.prepare(m)
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        for _ in range(3):
            runner(xd, idd, wd)
    torch.accelerator.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        runner(xd, idd, wd)
    torch.accelerator.synchronize(device)
    for _ in range(WARMUP):
        graph.replay()
    torch.accelerator.synchronize(device)
    samples = []
    for _ in range(BATCHES):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(REPLAYS_PER_BATCH):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / REPLAYS_PER_BATCH)  # ms per call
    samples_sorted = sorted(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "p10_ms": samples_sorted[int(0.1 * (len(samples) - 1))],
        "p90_ms": samples_sorted[int(0.9 * (len(samples) - 1))],
        "variance_ms2": statistics.variance(samples),
        "n_batches": BATCHES,
        "replays_per_batch": REPLAYS_PER_BATCH,
    }


def environment() -> dict:
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(
            {"gpu": p.name, "sm": f"{p.major}.{p.minor}", "cuda": torch.version.cuda}
        )
    try:
        import triton  # noqa: F401

        env["triton"] = triton.__version__
    except Exception:
        env["triton"] = None
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--num-experts", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backends", default="native,native_apply")
    ap.add_argument("--sizes", default="1,2,4,8,16,64,256,1024")
    ap.add_argument("--gemv-rows", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--rtol", type=float, default=0.03)
    ap.add_argument("--atol", type=float, default=0.0002)
    ap.add_argument("--no-timing", action="store_true", help="correctness stage only")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    bank, manifest = load_layer_bank(a.shard, a.prefix, a.num_experts)
    manifest["environment"] = environment()
    manifest["args"] = vars(a)
    manifest["tolerances_note"] = (
        "atol/rtol are prior candidates; a FAIL is investigated, never relaxed"
    )
    write_manifest(manifest, out / "manifest.json")
    hidden = bank["w13_weight"].shape[2] * 2
    sizes = [int(s) for s in a.sizes.split(",")]
    runners = {}
    for name in a.backends.split(","):
        if name == "native":
            runners[name] = NativeRunner(bank, device, gemv_rows=a.gemv_rows)
        elif name == "native_apply":
            runners[name] = ExpertsApplyRunner(bank, device, gemv_rows=a.gemv_rows)
        elif name == "marlin":
            runners[name] = MarlinRunner(bank, device)
    correctness = {}
    with open(out / "timing.jsonl", "w") as tl, open(out / "summary.csv", "w") as sc:
        sc.write("backend,M,path,median_ms,p10_ms,p90_ms,variance_ms2,correct\n")
        for m in sizes:
            x, ids, w = make_inputs(m, hidden, a.num_experts, a.seed + m)
            (out / "inputs").mkdir(exist_ok=True)
            torch.save({"x": x, "ids": ids, "w": w}, out / "inputs" / f"m{m}.pt")
            inputs_sha = {
                "x": sha256_tensor(x),
                "ids": sha256_tensor(ids),
                "w": sha256_tensor(w),
            }
            for name, runner in runners.items():
                c = check_correctness(runner, bank, m, x, ids, w, a.atol, a.rtol)
                c["inputs_sha256"] = inputs_sha
                c["path"] = runner.path(m) if hasattr(runner, "path") else name
                correctness[f"{name}/M{m}"] = c
                (out / "correctness.json").write_text(
                    json.dumps(correctness, indent=1) + "\n"
                )
                if not c["pass"]:
                    v = c["vs_source_f32_globals"]
                    print(
                        f"{name} M={m}: correctness FAIL "
                        f"({v['violations']}/{v['elements']}, "
                        f"sticky={c['sticky_error']}, "
                        f"non_finite={c['non_finite']}); no timing"
                    )
                    sc.write(f"{name},{m},{c['path']},,,,,FAIL\n")
                    continue
                if a.no_timing:
                    sc.write(f"{name},{m},{c['path']},,,,,PASS\n")
                    continue
                t = time_graph(runner, m, x, ids, w)
                rec = {
                    "backend": name,
                    "M": m,
                    "path": c["path"],
                    "time": time.time(),
                    **t,
                }
                tl.write(json.dumps(rec) + "\n")
                sc.write(
                    f"{name},{m},{c['path']},{t['median_ms']:.4f},{t['p10_ms']:.4f},{t['p90_ms']:.4f},{t['variance_ms2']:.6f},PASS\n"
                )
                print(f"{name} M={m} {c['path']}: median {t['median_ms']:.4f} ms")
    (out / "correctness.json").write_text(json.dumps(correctness, indent=1) + "\n")
    (out / "command.txt").write_text(" ".join(os.sys.argv) + "\n")


if __name__ == "__main__":
    main()
