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
      [--backends native,native_kernel,marlin] [--sizes 1,2,4,8,16,64,256,1024]
      [--patterns uniform,working_set] [--working-set-size 32]

Backends: "native" calls the kernels with independently allocated
workspaces (package-level pre-check); "native_kernel" runs
NativeNvFp4Experts inside the real FusedMoEKernel with the worker's
WorkspaceManager (the model path; the manager is locked before capture);
"marlin" is the reference backend. Route patterns (uniform, fixed shared
working set) are reported separately. Backends of one shape are captured
together and timed with the order reversed every other batch. Correctness
is reported against the source-semantics oracle (float32 globals) and, as
a second column, against an oracle that uses the backend's float16 per-row
globals; the pass verdict uses the source column.
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


PATTERNS = ("uniform", "working_set")


def make_inputs(
    m: int,
    hidden: int,
    num_experts: int,
    seed: int,
    pattern: str = "uniform",
    working_set: int = 32,
    working_set_seed: int = 0,
):
    """Synthetic inputs (not model-observed activations).

    Routes are distinct within a token; duplicates across tokens allowed.
    "uniform": every token draws from all experts. "working_set": every
    token draws from one fixed small set of `working_set` experts chosen once
    from `working_set_seed`, the same set for every M.
    """
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn((m, hidden), generator=g) * 0.5).to(torch.bfloat16)
    if pattern == "uniform":
        pool = torch.arange(num_experts)
    elif pattern == "working_set":
        if not TOP_K <= working_set <= num_experts:
            raise ValueError("working_set must be in [TOP_K, num_experts]")
        gws = torch.Generator().manual_seed(working_set_seed)
        pool = torch.randperm(num_experts, generator=gws)[:working_set]
    else:
        raise ValueError(f"unknown route pattern {pattern!r}")
    ids = torch.stack(
        [pool[torch.randperm(len(pool), generator=g)[:TOP_K]] for _ in range(m)]
    ).to(torch.int32)
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
        self._ws = {}
        self._m = None

    @property
    def error(self):
        """Sticky error flag of the workspace used by the last/next call."""
        if self._m is None:
            return None
        return self.prepare(self._m).error

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
        self._m = x.shape[0]
        ws = self.prepare(x.shape[0])
        if x.shape[0] <= self.gemv_rows:
            return self.nb.gemv(x, w, ids, self.bank, self.step_map, ws)
        return self.npf.prefill(x, w, ids, self.bank, self.step_map, ws)


class KernelRunner(NativeRunner):
    """NativeNvFp4Experts inside the real FusedMoEKernel with the worker's
    WorkspaceManager: workspace_shapes() -> get_simultaneous() -> apply()
    carving, exactly the model path. The manager is grown during prepare()
    and locked before capture, so a graph that needed more scratch would
    fail to capture instead of silently allocating.
    """

    name = "native_kernel"

    def __init__(self, bank, device, gemv_rows: int = 1):
        super().__init__(bank, device, gemv_rows)
        from types import SimpleNamespace

        import vllm.model_executor.layers.fused_moe.modular_kernel as mk
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
            MoEPrepareAndFinalizeNoDPEPModular,
        )
        from vllm.model_executor.layers.quantization.nvfp4_native.experts import (
            NativeNvFp4Experts,
        )
        from vllm.v1.worker import workspace as ws

        self._ws = ws
        self._act = MoEActivation.SILU
        self.experts = NativeNvFp4Experts.__new__(NativeNvFp4Experts)
        self.experts.moe_config = SimpleNamespace(
            experts_per_token=TOP_K, max_num_tokens=1024, moe_parallel_config=None
        )
        self.experts.quant_config = SimpleNamespace(
            gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None, a2_scale=None
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
        if not ws.is_workspace_manager_initialized():
            ws.init_workspace_manager(device)
        self.kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(), self.experts
        )
        self._prepared: set[int] = set()

    @property
    def error(self):
        return self.experts._error

    def workspace_bytes(self) -> int:
        manager = self._ws.current_workspace_manager()
        return sum(
            manager._workspace_size_bytes(w) for w in manager._current_workspaces
        )

    def prepare(self, m: int):
        if m in self._prepared:
            return None
        hidden = self.bank["w13_weight"].shape[2] * 2
        x = torch.zeros((m, hidden), dtype=torch.bfloat16, device=self.device)
        ids = torch.arange(TOP_K, device=self.device, dtype=torch.int32).repeat(m, 1)
        w = torch.full((m, TOP_K), 1.0 / TOP_K, device=self.device)
        self._ws.unlock_workspace()
        self(x, ids, w)  # grows the manager's buffer for this shape
        torch.accelerator.synchronize(self.device)
        self._ws.lock_workspace()
        self._prepared.add(m)
        return None

    def __call__(self, x, ids, w):
        return self.kernel.apply(
            hidden_states=x,
            w1=self.bank["w13_weight"],
            w2=self.bank["w2_weight"],
            topk_weights=w,
            topk_ids=ids,
            activation=self._act,
            global_num_experts=self.rows,
            expert_map=None,
            apply_router_weight_on_input=False,
        )


class MarlinRunner:
    """Marlin NVFP4 MoE on the same raw bank after the Marlin repack.

    Replicates the normal loader's handling of the w13 global scales
    (column 0 is kept; gate != up is recorded, not fixed): see
    `marlin_globals` in the manifest. Uses upstream's
    prepare_moe_fp4_layer_for_marlin and fused_marlin_moe unchanged.
    """

    name = "marlin"

    def __init__(self, bank, device):
        from types import SimpleNamespace

        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
            fused_marlin_moe,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            prepare_moe_fp4_layer_for_marlin,
        )
        from vllm.scalar_type import scalar_types

        self._fused_marlin_moe = fused_marlin_moe
        self._quant_type_id = scalar_types.float4_e2m1f.id
        self.device = device
        rows, n2, kh = bank["w13_weight"].shape
        intermediate, hidden = n2 // 2, kh * 2
        self.rows = rows
        g13 = bank["w13_weight_scale_2"].to(torch.float32).reshape(rows, -1)
        if g13.shape[1] == 1:
            g13 = g13.repeat(1, 2)
        gate, up = g13[:, 0], g13[:, 1]
        diff = (gate - up).abs()
        nonzero = (gate != 0) & (up != 0)
        self.globals_report = {
            "experts_with_gate_ne_up": int((diff > 0).sum()),
            "max_abs_diff": float(diff.max()),
            "ratio_min": float((up[nonzero] / gate[nonzero]).min())
            if nonzero.any()
            else None,
            "ratio_max": float((up[nonzero] / gate[nonzero]).max())
            if nonzero.any()
            else None,
            "experts_with_a_zero": int((~nonzero).sum()),
            "conversion": "w13 global = column 0 (gate), as the ModelOpt loader does",
            "source_equivalent": bool(int((diff > 0).sum()) == 0),
        }
        layer = SimpleNamespace(
            moe_config=SimpleNamespace(
                num_local_experts=rows,
                hidden_dim=hidden,
                intermediate_size_per_partition=intermediate,
            ),
            params_dtype=torch.bfloat16,
            w13_weight=torch.nn.Parameter(
                bank["w13_weight"].to(device), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                bank["w2_weight"].to(device), requires_grad=False
            ),
            w13_weight_scale=torch.nn.Parameter(
                bank["w13_weight_scale"].to(device), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(
                bank["w2_weight_scale"].to(device), requires_grad=False
            ),
            w13_weight_scale_2=torch.nn.Parameter(
                gate.contiguous().to(device), requires_grad=False
            ),
            w2_weight_scale_2=torch.nn.Parameter(
                bank["w2_weight_scale_2"].to(torch.float32).reshape(rows).to(device),
                requires_grad=False,
            ),
        )
        prepare_moe_fp4_layer_for_marlin(layer)
        self.layer = layer
        self.error = None

    def path(self, m: int) -> str:
        return "marlin"

    def prepare(self, m: int):
        return None

    def __call__(self, x, ids, w):
        layer = self.layer
        return self._fused_marlin_moe(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            bias1=None,
            bias2=None,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            topk_weights=w,
            topk_ids=ids,
            quant_type_id=self._quant_type_id,
            global_num_experts=self.rows,
            expert_map=None,
            global_scale1=layer.w13_weight_scale_2,
            global_scale2=layer.w2_weight_scale_2,
            workspace=layer.workspace,
        )


def backend_row_globals(
    bank: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The backend loader's transformation of the global scales, replicated
    in plain torch so the bench needs no vllm import: w13 gate rows take
    column 0 and up rows column 1 (per-row float16), w2 one value per row."""
    e, n2, _ = bank["w13_weight"].shape
    hidden = bank["w2_weight"].shape[1]
    g13 = bank["w13_weight_scale_2"].to(torch.float32).reshape(e, -1)
    if g13.shape[1] == 1:
        g13 = g13.repeat(1, 2)
    half = n2 // 2
    w13_rows = torch.cat(
        (g13[:, :1].repeat(1, half), g13[:, 1:2].repeat(1, half)), dim=1
    )
    g2 = bank["w2_weight_scale_2"].to(torch.float32).reshape(e, -1)
    w2_rows = g2[:, :1].repeat(1, hidden) if g2.shape[1] == 1 else g2
    return w13_rows.to(torch.float16), w2_rows.to(torch.float16)


def _oracle(bank, x, ids, w, f16_globals: bool):
    g13, g2 = bank["w13_weight_scale_2"], bank["w2_weight_scale_2"]
    if f16_globals:
        g13, g2 = backend_row_globals(bank)
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
        # The worst elements, so a handful of violations can be characterised
        # (near-cancellation vs systematic) without re-running.
        "top_violations": _top_violations(got, ref, diff, tol),
    }


def _top_violations(got, ref, diff, tol, limit: int = 8) -> list[dict]:
    excess = (diff - tol).reshape(-1)
    n = min(limit, int((excess > 0).sum()))
    if n == 0:
        return []
    idx = torch.topk(excess, n).indices
    g, r, d, t = (x.reshape(-1)[idx] for x in (got, ref, diff, tol))
    return [
        {
            "flat_index": int(i),
            "got": float(gv),
            "ref": float(rv),
            "abs_diff": float(dv),
            "tol": float(tv),
        }
        for i, gv, rv, dv, tv in zip(idx.tolist(), g, r, d, t)
    ]


def check_correctness(runner, m, x, ids, w, ref_source, ref_backend, atol, rtol):
    if hasattr(runner, "_m"):
        runner._m = m
    if getattr(runner, "error", None) is not None:
        runner.error.zero_()
    out = runner(x.to(runner.device), ids.to(runner.device), w.to(runner.device))
    torch.accelerator.synchronize(runner.device)
    got = out.detach().cpu().float()
    source = _compare(got, ref_source, atol, rtol)
    backend_globals = _compare(got, ref_backend, atol, rtol)
    err = getattr(runner, "error", None)
    sticky = int(err.item()) if err is not None else None
    return {
        "vs_source_f32_globals": source,
        "vs_backend_f16_globals": backend_globals,
        "sticky_error": sticky,
        "non_finite": int((~torch.isfinite(got)).sum()),
        "pass": source["pass"]
        and sticky in (None, 0)
        and bool(torch.isfinite(got).all()),
        "output_sha256": sha256_tensor(out),
        "_got": got,
    }


class CapturedCall:
    """One backend's MoE call captured in a CUDA graph, with its live output."""

    def __init__(self, runner, m, x, ids, w, reference, atol, rtol):
        self.runner, self.reference, self.atol, self.rtol = (
            runner,
            reference,
            atol,
            rtol,
        )
        device = runner.device
        # The graph replays read these addresses; keep the tensors alive for
        # the object's lifetime (the graph pool does not own external inputs,
        # and several backends are captured per shape).
        self.inputs = (x.to(device), ids.to(device), w.to(device))
        xd, idd, wd = self.inputs
        runner.prepare(m)
        stream = torch.cuda.Stream(device)
        # Input copies were enqueued on the current stream; order them first.
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                runner(xd, idd, wd)
        torch.accelerator.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.captured = runner(xd, idd, wd)
        torch.accelerator.synchronize(device)

    def replay_matches(self) -> dict:
        got = self.captured.detach().cpu().float()
        cmp = _compare(got, self.reference.float(), self.atol, self.rtol)
        err = getattr(self.runner, "error", None)
        cmp["sticky_error"] = int(err.item()) if err is not None else None
        cmp["non_finite"] = int((~torch.isfinite(got)).sum())
        cmp["pass"] = (
            cmp["pass"] and cmp["sticky_error"] in (None, 0) and cmp["non_finite"] == 0
        )
        return cmp


def time_interleaved(calls: dict[str, CapturedCall]) -> dict[str, dict]:
    """Time several captured calls of one shape with alternating order.

    Each of the BATCHES rounds times one batch of REPLAYS_PER_BATCH replays
    per backend; the backend order is reversed on every other round so no
    backend is always measured first (thermal/order bias). Outputs stay live
    and are checked outside the timed region, after warmup and after timing.
    """
    names = list(calls)
    for c in calls.values():
        for _ in range(WARMUP):
            c.graph.replay()
    torch.accelerator.synchronize(next(iter(calls.values())).runner.device)
    after_warmup = {n: calls[n].replay_matches() for n in names}
    samples: dict[str, list[float]] = {n: [] for n in names}
    order_log: list[list[str]] = []
    for i in range(BATCHES):
        order = names if i % 2 == 0 else list(reversed(names))
        order_log.append(list(order))
        for n in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(REPLAYS_PER_BATCH):
                calls[n].graph.replay()
            end.record()
            end.synchronize()
            samples[n].append(start.elapsed_time(end) / REPLAYS_PER_BATCH)
    after_timing = {n: calls[n].replay_matches() for n in names}
    results = {}
    for n in names:
        ss = sorted(samples[n])
        valid = after_warmup[n]["pass"] and after_timing[n]["pass"]
        results[n] = {
            "valid": valid,
            "replay_check_after_warmup": after_warmup[n],
            "replay_check_after_timing": after_timing[n],
            "samples_ms": samples[n],
            "median_ms": statistics.median(samples[n]) if valid else None,
            "p10_ms": ss[int(0.1 * (len(ss) - 1))] if valid else None,
            "p90_ms": ss[int(0.9 * (len(ss) - 1))] if valid else None,
            "variance_ms2": statistics.variance(samples[n]) if valid else None,
            "n_batches": BATCHES,
            "replays_per_batch": REPLAYS_PER_BATCH,
            "interleaved_with": [o for o in names if o != n],
            "batch_order": order_log,
        }
    return results


def time_graph(runner, m, x, ids, w, reference: torch.Tensor, atol: float, rtol: float):
    """Single-backend convenience wrapper around CapturedCall/time_interleaved."""
    call = CapturedCall(runner, m, x, ids, w, reference, atol, rtol)
    return time_interleaved({runner.name: call})[runner.name]


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
    ap.add_argument("--backends", default="native,native_kernel")
    ap.add_argument("--patterns", default=",".join(PATTERNS))
    ap.add_argument("--working-set-size", type=int, default=32)
    ap.add_argument("--sizes", default="1,2,4,8,16,64,256,1024")
    ap.add_argument("--gemv-rows", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--rtol", type=float, default=0.03)
    ap.add_argument("--atol", type=float, default=0.0002)
    ap.add_argument("--no-timing", action="store_true", help="correctness stage only")
    ap.add_argument(
        "--save-outputs",
        action="store_true",
        help="save each backend's eager output per shape under outputs/",
    )
    ap.add_argument(
        "--device", default="cuda", help="cuda (default); cpu only for the mock test"
    )
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
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
        elif name == "native_kernel":
            runners[name] = KernelRunner(bank, device, gemv_rows=a.gemv_rows)
        elif name == "marlin":
            runners[name] = MarlinRunner(bank, device)
    if "marlin" in runners:
        manifest["marlin_globals"] = runners["marlin"].globals_report
        write_manifest(manifest, out / "manifest.json")
    correctness = {}
    patterns = [p for p in a.patterns.split(",") if p]
    with open(out / "timing.jsonl", "w") as tl, open(out / "summary.csv", "w") as sc:
        sc.write(
            "backend,pattern,M,path,median_ms,p10_ms,p90_ms,variance_ms2,"
            "correct,source_equivalent\n"
        )
        for pattern in patterns:
            for m in sizes:
                kwargs = (
                    {}
                    if pattern == "uniform"
                    else {
                        "pattern": pattern,
                        "working_set": a.working_set_size,
                        "working_set_seed": a.seed,
                    }
                )
                x, ids, w = make_inputs(m, hidden, a.num_experts, a.seed + m, **kwargs)
                (out / "inputs").mkdir(exist_ok=True)
                torch.save(
                    {"x": x, "ids": ids, "w": w, "pattern": pattern},
                    out / "inputs" / f"{pattern}_m{m}.pt",
                )
                inputs_sha = {
                    "x": sha256_tensor(x),
                    "ids": sha256_tensor(ids),
                    "w": sha256_tensor(w),
                }
                # Source-semantics oracle (pass/fail and the graph-replay
                # reference) and the backend-globals oracle (second column),
                # once per shape.
                ref_source = _oracle(bank, x, ids, w, False).float()
                ref_backend = _oracle(bank, x, ids, w, True).float()
                timeable: dict[str, tuple] = {}
                for name, runner in runners.items():
                    # Shapes are prepared before any eager call: a runner that
                    # locks its WorkspaceManager after capture must have grown
                    # the buffer for this M first.
                    runner.prepare(m)
                    c = check_correctness(
                        runner, m, x, ids, w, ref_source, ref_backend, a.atol, a.rtol
                    )
                    if a.save_outputs:
                        (out / "outputs").mkdir(exist_ok=True)
                        torch.save(
                            {"got": c.pop("_got"), "ref_source": ref_source},
                            out / "outputs" / f"{name}_{pattern}_m{m}.pt",
                        )
                    else:
                        c.pop("_got", None)
                    c["inputs_sha256"] = inputs_sha
                    c["pattern"] = pattern
                    c["path"] = runner.path(m) if hasattr(runner, "path") else name
                    # A backend whose weight conversion is not source-equivalent
                    # (Marlin with gate != up globals) is still checked, but its
                    # rows are marked so they are never read as a
                    # same-arithmetic comparison.
                    equivalent = getattr(runner, "globals_report", {}).get(
                        "source_equivalent", True
                    )
                    c["source_equivalent"] = equivalent
                    eq = "yes" if equivalent else "NO"
                    correctness[f"{name}/{pattern}/M{m}"] = c
                    (out / "correctness.json").write_text(
                        json.dumps(correctness, indent=1) + "\n"
                    )
                    if not c["pass"]:
                        v = c["vs_source_f32_globals"]
                        print(
                            f"{name} {pattern} M={m}: correctness FAIL "
                            f"({v['violations']}/{v['elements']}, "
                            f"sticky={c['sticky_error']}, "
                            f"non_finite={c['non_finite']}); no timing"
                        )
                        sc.write(f"{name},{pattern},{m},{c['path']},,,,,FAIL,{eq}\n")
                        continue
                    if a.no_timing:
                        sc.write(f"{name},{pattern},{m},{c['path']},,,,,PASS,{eq}\n")
                        continue
                    timeable[name] = (runner, c, eq)
                if not timeable:
                    continue
                calls = {
                    name: CapturedCall(r, m, x, ids, w, ref_source, a.atol, a.rtol)
                    for name, (r, _c, _eq) in timeable.items()
                }
                timings = time_interleaved(calls)
                for name, (runner, c, eq) in timeable.items():
                    t = timings[name]
                    rec = {
                        "backend": name,
                        "pattern": pattern,
                        "M": m,
                        "path": c["path"],
                        "time": time.time(),
                        "source_equivalent": c["source_equivalent"],
                        "workspace_bytes": (
                            runner.workspace_bytes()
                            if hasattr(runner, "workspace_bytes")
                            else None
                        ),
                        **t,
                    }
                    tl.write(json.dumps(rec) + "\n")
                    if not t["valid"]:
                        print(
                            f"{name} {pattern} M={m}: graph replay check FAIL; "
                            "timing invalidated"
                        )
                        sc.write(
                            f"{name},{pattern},{m},{c['path']},,,,,REPLAY_FAIL,{eq}\n"
                        )
                        continue
                    sc.write(
                        f"{name},{pattern},{m},{c['path']},{t['median_ms']:.4f},"
                        f"{t['p10_ms']:.4f},{t['p90_ms']:.4f},"
                        f"{t['variance_ms2']:.6f},PASS,{eq}\n"
                    )
                    print(
                        f"{name} {pattern} M={m} {c['path']}: "
                        f"median {t['median_ms']:.4f} ms"
                    )
    (out / "correctness.json").write_text(json.dumps(correctness, indent=1) + "\n")
    (out / "command.txt").write_text(" ".join(os.sys.argv) + "\n")


if __name__ == "__main__":
    main()
