# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exclusive GPU-hot / pinned-CPU-cold NVFP4 Marlin tier adapter.

Pinned to vLLM 1970f3ed4; unsharded FlashNext only, without torch.compile.
Policy is the existing expert-tier heat/periodic migration policy, not demand
caching.

CUDA Graph contract (FULL decode graphs, eager prefill):

- Everything a captured graph reads or writes lives at a fixed address: hot
  weights, pinned cold banks (through UVA), the two expert maps, and the
  per-layer routing record buffer. All of these are updated in place.
- The forward only records routing into that buffer; no host copy, policy
  decision, or migration happens inside a layer. A replay executes none of
  this Python, so the runner's post-forward hook (`finish_model_forward`)
  performs the single D2H copy, heat update, RAM TEMP swaps, and map
  publication after every forward, replayed or eager.
- Migration runs on the runner's stream and waits for completion before the
  next forward is issued, so weights and maps never change under a replay.
"""

from __future__ import annotations

import atexit
import gc
import json
import logging
import math
import os
import threading
import time
import weakref
from collections import Counter
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

from .async_migration import SpareRing
from .tier_policy import TierPolicy

LOGGER = logging.getLogger(__name__)
PREFIX = "VLLM_LAB_EXPERT_TIER_"
TENSORS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
)
SCALE_PROPERTIES = {
    "w1_scale": "w13_weight_scale",
    "w2_scale": "w2_weight_scale",
    "g1_alphas": "w13_weight_scale_2",
    "g2_alphas": "w2_weight_scale_2",
}
VERIFY_RTOL, VERIFY_ATOL = 2e-2, 2e-2
SPLIT_MODES = ("fused", "modular")
MAX_SPEC_ROWS = 8
# moe_align_block_size histograms by *mapped* id in a buffer of its
# num_experts argument (+1) entries, so a mapped row must stay below the
# logical expert count; a bank with more physical rows than experts is
# aligned by logical id and its blocks are mapped to rows afterwards.
NATIVE_KERNEL = SimpleNamespace(fused_experts="native")
_NATIVE_WORKSPACES: dict[tuple[str, int], Any] = {}
_NATIVE_PREFILL_WORKSPACES: dict[tuple[str, int], Any] = {}
_PREFILL_SCRATCH: dict[tuple[Any, ...], Any] = {}
_CAPTURE_COUNT = 0
# Never attach CPU owners to Parameter.__dict__: reload metadata copies it.
_CPU_SOURCES: dict[int, tuple[weakref.ReferenceType[Any], Any]] = {}


@dataclass(frozen=True)
class Settings:
    capacity_bytes: int
    stats_every: int = 256
    verify_init: bool = True
    sync_tokens: int = 50
    swaps_per_token: float = 1.0
    decay: float = 0.999
    hysteresis: float = 1.3
    dwell_tokens: int = 0
    max_swaps_per_resync: int = 0
    # Pinned RAM TEMP rows: the largest wave of slot-independent swaps that
    # one resync moves with two stream waits instead of two per swap.
    temp_slots: int = 8
    # "fused": both partitions' Marlin GEMM chains write disjoint rows of one
    # per-slot buffer that is reduced once. "modular": two stock modular
    # kernel calls, clone, and add (the original path, kept for fallback).
    split: str = "fused"
    # "records": static routing records read back once per forward. Other
    # names resolve through OBSERVERS (device-side observers register there).
    observer: str = "records"
    # Exchange rows on a migration stream overlapped with the next forward,
    # flipping logical-to-physical row tables at the following boundary.
    # temp_slots spare VRAM and RAM rows per layer, charged to the budget.
    async_migration: bool = False
    # Decode-time VRAM staging of the selected cold experts: top_k spare rows
    # per layer at the end of the hot bank, charged to the capacity budget.
    # Off by default: correct on the GPU but 7.7% slower than the fused
    # two-partition path in its first measurement; launch volume and the
    # lower resident hit rate are both candidate causes, not yet separated.
    staging: bool = False
    # "uniform", or one explicit hot slot count per layer (48 integers).
    # Static for the process: bank sizes fix the addresses captured graphs
    # read. A global, demand-driven pool like FreeToken's is separate work.
    layer_slots: str = "uniform"
    # Promote mode: per-token device LRU owns the placement (FreeToken-style
    # miss fill); the heat policy only observes. Needs staging; excludes the
    # asynchronous exchange path (its spare rings become the promote rings).
    promote: bool = False
    # "reference" (torch, host-synchronizing; tests and non-CUDA devices) or
    # "device" (vllm._lab_expert_tier.device_lru, graph-capturable).
    planner: str = "device"
    # RAM backing (promote mode only): keep the loader's pinned UVA source
    # of every expert as the RAM bank instead of compacting the cold rows
    # into a new allocation. RAM row e always holds expert e, so an eviction
    # flips the tables and never writes back (no D2H, no RAM pool, no
    # shadows). Host bytes: the full source stays resident (the loader's
    # own peak); no second copy of any bank is made.
    ram_backing: bool = False
    # Global pool (promote + RAM backing only): one VRAM bank shared by all
    # layers with one LRU over every (layer, expert), FreeToken style, so
    # the resident count per layer floats. Staging rows are shared too.
    global_pool: bool = False
    # Expert kernel: "marlin" (vLLM Marlin on repacked banks) or "native"
    # (FreeToken-derived GEMV on the checkpoint layout, `native_nvfp4`;
    # needs RAM_BACKING=1 and SPLIT=fused; the loader keeps the raw banks).
    moe_kernel: str = "marlin"
    # Native multi-token path: "gemv" runs the decode GEMV per route (the
    # adapter's fallback); "grouped" calls native_prefill.prefill (grouped
    # GEMM over the same bank and workspace). Decode is unaffected.
    native_prefill: str = "gemv"
    # Rows at or below this count take the decode GEMV even when the
    # multi-token path is "grouped": a speculative verify step has
    # 1 + k <= spec_rows rows and is decode, not prefill. Defaults to
    # spec_rows (1 without speculation, so the plain decode step only).
    native_gemv_rows: int = 1
    # Staged prefill: rows of a device scratch bank (shared by all layers,
    # used one layer at a time). A multi-token forward copies the cold experts
    # it routes to into the scratch with the bank copy kernel (contiguous
    # rows, host -> device) and runs the kernel on device memory instead of
    # gathering host rows through UVA inside the kernel. Experts beyond the
    # scratch rows stay on the UVA partition. 0 = off (default).
    prefill_stage_rows: int = 0
    # Fused per-layer routing record (device observer only): one program
    # writes the observer records, counts, route totals, and the sticky
    # error (including the former device assertion) instead of ~30 small
    # kernels per layer. Rows beyond the fused width take the old path.
    record_kernel: bool = False
    # Shared-expert gate of the Qwen4 exp MoE block: "torch" (cuBLAS dot,
    # sigmoid, multiply) or "fused" (one Triton program per token,
    # `shared_gate.py`, FreeToken's gate kernel taken one step further).
    shared_gate: str = "torch"
    # Native decode output: "clone" copies the adapter's workspace output
    # (one copy per layer) or "alias" returns it directly. The alias is
    # consumed within the layer by the MoE runner's out-of-place
    # `shared_output + fused_output` (or by the next layer's hyper-connection
    # combine without a shared expert), before the next layer's gemv
    # rewrites the workspace; the two-partition path keeps its own copy.
    native_output: str = "clone"
    # Expert row copy launch shape: "stripe" (current) or "chunks"
    # (FreeToken's fast_index_copy_multi shape: 8 programs x 32 warps per
    # bank over (row, chunk) pairs). See promote.COPY_SHAPES.
    copy_shape: str = "stripe"
    # Rows the pool decode fast path serves in one step (1 + speculative
    # tokens): staging holds spec_rows * top_k rows and the step program is
    # that wide, so a verify step with 1 + k <= spec_rows rows stays on the
    # captured decode path; wider batches take the eager path with every row
    # processed. Needs GLOBAL_POOL=1 when above 1. The tier admits vLLM
    # speculation only for method "ngram" with 1 + k <= spec_rows.
    spec_rows: int = 1
    # Global pool placement controls (device scalars, also settable at run
    # time through CONTROL_FILE): promotions per layer call (0 = unlimited),
    # promote only every N forwards, misses a key needs before promotion,
    # forwards a used row stays unevictable. See global_pool.set_control.
    promote_limit: int = 0
    promote_interval: int = 1
    promote_min_misses: int = 1
    protect_recent: int = 0
    # JSON file polled at every stats report: {"promote_limit": .., "gate": ..}
    # (any subset of global_pool.CONTROL_FIELDS); validated whole, applied at
    # the forward boundary, invalid or partial content keeps the last values.
    control_file: str = ""
    # Steady-state pool verification trigger: a file whose mtime change makes
    # the next forward boundary compare every resident expert row in the pool
    # bank with its host source row (bytes) and log the mismatches. Needs
    # RAM_BACKING=1 (source row = expert id). "" = off (default).
    verify_file: str = ""
    # Copy launch grid: programs per bank and int32 words per iteration.
    copy_programs: int = 0  # 0 = the shape's default (32 stripe / 8 chunks)
    copy_words: int = 4096
    # Optional whole-device check: tier bytes + the draft reservation must
    # fit this many GiB before the draft loads (0 = no check). Never a
    # source of bytes; the reservation comes from the checkpoint headers.
    vram_budget_gib: float = 0.0
    # Draft resident bytes may exceed the reservation by this fraction
    # (load-time buffers, alignment) before the post-load check fails.
    draft_tolerance: float = 0.05

    def policy_kwargs(self):
        # sync=0 freezes the initial partition, while heat/token credit still
        # accumulates. This is the original heat policy, never an LRU switch.
        return {
            "sync_period": self.sync_tokens,
            "swaps_per_token": self.swaps_per_token,
            "decay": self.decay,
            "hysteresis": self.hysteresis,
            "dwell_tokens": self.dwell_tokens,
            "max_swaps_per_resync": self.max_swaps_per_resync,
        }

    @classmethod
    def from_env(cls):
        known = {
            "GIB",
            "STATS_EVERY",
            "VERIFY_INIT",
            "SYNC_TOKENS",
            "SWAPS_PER_TOKEN",
            "DECAY",
            "HYSTERESIS",
            "DWELL_TOKENS",
            "MAX_SWAPS_PER_RESYNC",
            "TEMP_SLOTS",
            "SPLIT",
            "OBSERVER",
            "STAGING",
            "ASYNC_MIGRATION",
            "LAYER_SLOTS",
            "PROMOTE",
            "PLANNER",
            "RAM_BACKING",
            "GLOBAL_POOL",
            "MOE_KERNEL",
            "NATIVE_PREFILL",
            "NATIVE_GEMV_ROWS",
            "PREFILL_STAGE_ROWS",
            "RECORD_KERNEL",
            "SHARED_GATE",
            "NATIVE_OUTPUT",
            "COPY_SHAPE",
            "SPEC_ROWS",
            "PROMOTE_LIMIT",
            "PROMOTE_INTERVAL",
            "PROMOTE_MIN_MISSES",
            "PROTECT_RECENT",
            "CONTROL_FILE",
            "VERIFY_FILE",
            "COPY_PROGRAMS",
            "COPY_WORDS",
            "VRAM_BUDGET_GIB",
            "DRAFT_TOLERANCE",
        }
        unknown = {k[len(PREFIX) :] for k in os.environ if k.startswith(PREFIX)} - known
        if unknown:
            raise ValueError(f"Unknown expert tier settings: {sorted(unknown)}")
        value = Decimal(os.environ.get(PREFIX + "GIB", "0"))
        if not value.is_finite() or value < 0:
            raise ValueError("Expert tier GIB must be finite and nonnegative")
        if value == 0:
            if any(k.startswith(PREFIX) and k != PREFIX + "GIB" for k in os.environ):
                raise ValueError("Tier knobs require positive GIB")
            return None
        stats = int(os.environ.get(PREFIX + "STATS_EVERY", "256"))
        verify = os.environ.get(PREFIX + "VERIFY_INIT", "1")
        staging = os.environ.get(PREFIX + "STAGING", "0")
        asynchronous = os.environ.get(PREFIX + "ASYNC_MIGRATION", "0")
        promote = os.environ.get(PREFIX + "PROMOTE", "0")
        ram_backing = os.environ.get(PREFIX + "RAM_BACKING", "0")
        global_pool = os.environ.get(PREFIX + "GLOBAL_POOL", "0")
        flags = (verify, staging, asynchronous, promote, ram_backing, global_pool)
        if stats < 1 or any(flag not in ("0", "1") for flag in flags):
            raise ValueError(
                "STATS_EVERY must be positive; VERIFY_INIT, STAGING, "
                "ASYNC_MIGRATION, PROMOTE, RAM_BACKING, and GLOBAL_POOL must "
                "be 0 or 1"
            )
        if ram_backing == "1" and promote != "1":
            raise ValueError("RAM_BACKING requires PROMOTE=1")
        if global_pool == "1" and ram_backing != "1":
            raise ValueError("GLOBAL_POOL requires RAM_BACKING=1")
        if global_pool == "1" and os.environ.get(PREFIX + "SPLIT", "fused") != "fused":
            raise ValueError("GLOBAL_POOL requires SPLIT=fused")
        moe_kernel = os.environ.get(PREFIX + "MOE_KERNEL", "marlin")
        if moe_kernel not in ("marlin", "native"):
            raise ValueError("MOE_KERNEL must be marlin or native")
        if moe_kernel == "native" and ram_backing != "1":
            raise ValueError("MOE_KERNEL=native requires RAM_BACKING=1")
        record_kernel = os.environ.get(PREFIX + "RECORD_KERNEL", "0")
        if record_kernel not in ("0", "1"):
            raise ValueError("RECORD_KERNEL must be 0 or 1")
        shared_gate = os.environ.get(PREFIX + "SHARED_GATE", "torch")
        if shared_gate not in ("torch", "fused"):
            raise ValueError("SHARED_GATE must be torch or fused")
        native_output = os.environ.get(PREFIX + "NATIVE_OUTPUT", "clone")
        if native_output not in ("clone", "alias"):
            raise ValueError("NATIVE_OUTPUT must be clone or alias")
        copy_shape = os.environ.get(PREFIX + "COPY_SHAPE", "stripe")
        if copy_shape not in ("stripe", "chunks"):
            raise ValueError("COPY_SHAPE must be stripe or chunks")
        controls = {
            key: int(os.environ.get(PREFIX + key.upper(), default))
            for key, default in (
                ("promote_limit", "0"),
                ("promote_interval", "1"),
                ("promote_min_misses", "1"),
                ("protect_recent", "0"),
            )
        }
        from .global_pool import validate_control

        validate_control(controls)
        control_file = os.environ.get(PREFIX + "CONTROL_FILE", "").strip()
        verify_file = os.environ.get(PREFIX + "VERIFY_FILE", "").strip()
        if verify_file and ram_backing != "1":
            raise ValueError("VERIFY_FILE requires RAM_BACKING=1")
        copy_programs = int(os.environ.get(PREFIX + "COPY_PROGRAMS", "0"))
        copy_words = int(os.environ.get(PREFIX + "COPY_WORDS", "4096"))
        if copy_programs < 0:
            raise ValueError("COPY_PROGRAMS must be nonnegative (0 = default)")
        if copy_words < 32 or copy_words & (copy_words - 1):
            raise ValueError("COPY_WORDS must be a power of two >= 32")
        vram_budget_gib = float(os.environ.get(PREFIX + "VRAM_BUDGET_GIB", "0"))
        draft_tolerance = float(os.environ.get(PREFIX + "DRAFT_TOLERANCE", "0.05"))
        if vram_budget_gib < 0 or not 0 <= draft_tolerance <= 1:
            raise ValueError("VRAM_BUDGET_GIB >= 0 and 0 <= DRAFT_TOLERANCE <= 1")
        spec_rows = int(os.environ.get(PREFIX + "SPEC_ROWS", "1"))
        if not 1 <= spec_rows <= MAX_SPEC_ROWS:
            raise ValueError(f"SPEC_ROWS must be in [1, {MAX_SPEC_ROWS}]")
        if spec_rows > 1 and global_pool != "1":
            raise ValueError("SPEC_ROWS above 1 requires GLOBAL_POOL=1")
        native_prefill = os.environ.get(PREFIX + "NATIVE_PREFILL", "gemv")
        if native_prefill not in ("gemv", "grouped"):
            raise ValueError("NATIVE_PREFILL must be gemv or grouped")
        if native_prefill != "gemv" and moe_kernel != "native":
            raise ValueError("NATIVE_PREFILL requires MOE_KERNEL=native")
        native_gemv_rows = int(
            os.environ.get(PREFIX + "NATIVE_GEMV_ROWS", str(spec_rows))
        )
        if native_gemv_rows < 1:
            raise ValueError("NATIVE_GEMV_ROWS must be at least 1")
        prefill_stage_rows = int(os.environ.get(PREFIX + "PREFILL_STAGE_ROWS", "0"))
        if prefill_stage_rows < 0:
            raise ValueError("PREFILL_STAGE_ROWS must be nonnegative (0 = off)")
        if prefill_stage_rows and (ram_backing != "1" or moe_kernel != "native"):
            # Only the native chain reads every bank tensor (weights and
            # scales) from the partition it is handed; the Marlin chain keeps
            # scales on the layer, so a compact scratch would misindex them.
            raise ValueError(
                "PREFILL_STAGE_ROWS requires RAM_BACKING=1 and MOE_KERNEL=native"
            )
        planner = os.environ.get(PREFIX + "PLANNER", "device")
        if planner not in ("reference", "device"):
            raise ValueError("PLANNER must be reference or device")
        if promote == "1" and (staging != "1" or asynchronous == "1"):
            raise ValueError("PROMOTE requires STAGING=1 and ASYNC_MIGRATION=0")
        split = os.environ.get(PREFIX + "SPLIT", SPLIT_MODES[0])
        if split not in SPLIT_MODES:
            raise ValueError(f"SPLIT must be one of {SPLIT_MODES}")
        if moe_kernel == "native" and split != "fused":
            raise ValueError("MOE_KERNEL=native requires SPLIT=fused")
        observer = os.environ.get(PREFIX + "OBSERVER", "records")
        if not observer.isidentifier():
            raise ValueError("OBSERVER must be an observer registry name")
        if record_kernel == "1" and observer != "device":
            raise ValueError("RECORD_KERNEL requires OBSERVER=device")
        layer_slots = os.environ.get(PREFIX + "LAYER_SLOTS", "uniform").strip()
        if layer_slots != "uniform":
            try:
                counts = [int(item) for item in layer_slots.split(",")]
            except ValueError as error:
                raise ValueError("LAYER_SLOTS must be 'uniform' or integers") from error
            if not counts or any(count < 1 for count in counts):
                raise ValueError("LAYER_SLOTS entries must be positive integers")
            layer_slots = ",".join(str(count) for count in counts)
        integers = {
            key: int(os.environ.get(PREFIX + key, default))
            for key, default in (
                ("SYNC_TOKENS", "50"),
                ("DWELL_TOKENS", "0"),
                ("MAX_SWAPS_PER_RESYNC", "0"),
                ("TEMP_SLOTS", "8"),
            )
        }
        for key, integer in integers.items():
            if integer < 0 or (key == "TEMP_SLOTS" and integer < 1):
                raise ValueError(f"{key} must be a nonnegative integer")
        numbers = {
            key: float(os.environ.get(PREFIX + key, default))
            for key, default in (
                ("SWAPS_PER_TOKEN", "1"),
                ("DECAY", "0.999"),
                ("HYSTERESIS", "1.3"),
            )
        }
        for key, number in numbers.items():
            if (
                not math.isfinite(number)
                or number < 0
                or (key == "DECAY" and number > 1)
            ):
                raise ValueError(
                    f"{key} must be finite and nonnegative; DECAY must be <= 1"
                )
        return cls(
            int(value * (1 << 30)),
            stats,
            verify == "1",
            integers["SYNC_TOKENS"],
            numbers["SWAPS_PER_TOKEN"],
            numbers["DECAY"],
            numbers["HYSTERESIS"],
            integers["DWELL_TOKENS"],
            integers["MAX_SWAPS_PER_RESYNC"],
            integers["TEMP_SLOTS"],
            split,
            observer,
            asynchronous == "1",
            staging == "1",
            layer_slots,
            promote == "1",
            planner,
            ram_backing == "1",
            global_pool == "1",
            moe_kernel,
            native_prefill,
            native_gemv_rows,
            prefill_stage_rows,
            record_kernel == "1",
            shared_gate,
            native_output,
            copy_shape,
            spec_rows,
            controls["promote_limit"],
            controls["promote_interval"],
            controls["promote_min_misses"],
            controls["protect_recent"],
            control_file,
            verify_file,
            copy_programs,
            copy_words,
            vram_budget_gib,
            draft_tolerance,
        )


def capture_cpu_source(parameter, cpu_tensor):
    """Retain the *same* host allocation, never .cpu() a UVA tensor again."""
    global _CAPTURE_COUNT
    from .draft_scope import is_draft_load_scope

    if Settings.from_env() is not None and not is_draft_load_scope():
        identity = id(parameter)

        def release(dead_ref, identity=identity):
            current = _CPU_SOURCES.get(identity)
            # A late callback must not delete a replacement entry/id reuse.
            if current is not None and current[0] is dead_ref:
                _CPU_SOURCES.pop(identity)

        _CPU_SOURCES[identity] = (weakref.ref(parameter, release), cpu_tensor)
        _CAPTURE_COUNT += 1
        if _CAPTURE_COUNT % 32 == 0:
            LOGGER.warning(
                "LAB_EXPERT_TIER_HOST %s",
                json.dumps(
                    {
                        "captures": _CAPTURE_COUNT,
                        **_host_allocator_stats(),
                    },
                    sort_keys=True,
                ),
            )


def _get_cpu_source(parameter):
    entry = _CPU_SOURCES.get(id(parameter))
    if entry is None or entry[0]() is not parameter:
        return None
    return entry[1]


def _host_allocator_stats():
    """Read allocator counters only; never initialize CUDA or synchronize."""
    import torch

    stats = torch.cuda.memory.host_memory_stats()
    return {
        "cuda_initialized": torch.cuda.is_initialized(),
        "allocated_bytes": stats.get("allocated_bytes.current"),
        "active_bytes": stats.get("active_bytes.current"),
    }


def validate_mapping(mapping, required, slots):
    occupied = [slot for slot in mapping if slot >= 0]
    if any(slot >= slots for slot in occupied) or len(occupied) != len(set(occupied)):
        raise AssertionError("Cache mapping aliases slots or exceeds capacity")
    if any(slot < -1 for slot in mapping):
        raise AssertionError("Invalid negative cache mapping")
    for expert in required:
        if not 0 <= expert < len(mapping) or mapping[expert] < 0:
            raise AssertionError(f"Required expert {expert} has no GPU cache slot")


def uniform_slots(capacity_bytes, rows, num_experts, reserve=0):
    """Hot slots per layer; `reserve` staging rows per layer come first."""
    if not rows or len(set(rows)) != 1 or rows[0] <= 0:
        raise ValueError("Only uniform positive expert row sizes are supported")
    if reserve < 0:
        raise ValueError("Staging reserve must be nonnegative")
    slots = min(num_experts, capacity_bytes // sum(rows) - reserve)
    if slots < 1:
        raise ValueError("Capacity cannot hold one expert in every layer")
    return slots, (slots + reserve) * sum(rows)


def allocate_slots(capacity_bytes, rows, num_experts, reserve=0, layer_slots=None):
    """Hot slots for every layer under one exact byte budget.

    `layer_slots` is None for the uniform split, or one explicit slot count
    per layer. Explicit counts must each leave a hot and a cold partition
    and together fit the budget (each layer also carries `reserve` rows);
    nothing is rescaled silently. Returns (slots per layer, GPU bytes).
    """
    if layer_slots is None:
        slots, size = uniform_slots(capacity_bytes, rows, num_experts, reserve)
        return [slots] * len(rows), size
    if len(layer_slots) != len(rows):
        raise ValueError("LAYER_SLOTS must list one hot slot count per layer")
    if any(not 0 < slots < num_experts for slots in layer_slots):
        raise ValueError("Every layer needs a nonempty hot and cold partition")
    size = sum((slots + reserve) * row for slots, row in zip(layer_slots, rows))
    if size > capacity_bytes:
        raise ValueError("LAYER_SLOTS exceeds the capacity budget")
    return list(layer_slots), size


def check_verify_capacity(slots_per_layer, num_experts, top_k):
    """Init verification routes top-k rows into each partition of every layer.

    Reject an allocation that cannot host that before any GPU bank exists,
    naming the layer; verification is never skipped silently.
    """
    for index, slots in enumerate(slots_per_layer):
        if min(slots, num_experts - slots) < top_k:
            raise ValueError(
                f"Layer {index}: {slots} hot slots leave a partition smaller "
                f"than top-k {top_k}; init verification cannot run"
            )


def _check_kernel_scales(kernel, tensors):
    # Quant config holds tensor references. A config built before swapping to
    # cache tensors would silently continue reading full-source scales.
    experts = kernel.fused_experts
    for prop, name in SCALE_PROPERTIES.items():
        actual, expected = getattr(experts, prop), tensors[name]
        if actual is None or actual.data_ptr() != expected.data_ptr():
            raise RuntimeError(f"Stale or missing quant config reference: {name}")
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise RuntimeError(f"Quant config shape/dtype mismatch: {name}")


def validate_partition(hot_map, cold_map, hot_slots, cold_slots):
    if len(hot_map) != hot_slots + cold_slots or len(cold_map) != len(hot_map):
        raise AssertionError("Partition maps must cover the entire expert space")
    validate_mapping(hot_map, (), hot_slots)
    validate_mapping(cold_map, (), cold_slots)
    if sorted(v for v in hot_map if v >= 0) != list(range(hot_slots)):
        raise AssertionError("Every hot slot must have exactly one owner")
    if sorted(v for v in cold_map if v >= 0) != list(range(cold_slots)):
        raise AssertionError("Every cold slot must have exactly one owner")
    if any((h >= 0) == (c >= 0) for h, c in zip(hot_map, cold_map)):
        raise AssertionError("Every expert must reside in exactly one partition")


def maps_after_swap(hot_map, cold_map, old_expert, new_expert, hot_slot, cold_slot):
    if (
        hot_map[old_expert] != hot_slot
        or cold_map[new_expert] != cold_slot
        or cold_map[old_expert] != -1
        or hot_map[new_expert] != -1
    ):
        raise AssertionError("Swap does not match current physical placement")
    hot, cold = list(hot_map), list(cold_map)
    hot[old_expert], hot[new_expert] = -1, hot_slot
    cold[old_expert], cold[new_expert] = cold_slot, -1
    validate_partition(hot, cold, sum(v >= 0 for v in hot), sum(v >= 0 for v in cold))
    return tuple(hot), tuple(cold)


def swap_tensor_rows_wave(items, synchronize):
    """Original RAM TEMP order for a wave of slot-independent swaps.

    Each item is (hot, cold, temporary_row, hot_slot, cold_slot) and every
    item must own a distinct TEMP row and distinct slots. All D2H hot->TEMP
    copies complete before any H2D cold->hot; all H2D complete before any
    cold CPU storage is overwritten with TEMP. All six tensors move. The
    caller poisons on any partial-copy failure.
    """
    for hot, _, temporary, hot_slot, _ in items:
        for name in TENSORS:
            temporary[name].copy_(hot[name][hot_slot], non_blocking=True)
    synchronize()
    for hot, cold, _, hot_slot, cold_slot in items:
        for name in TENSORS:
            hot[name][hot_slot].copy_(cold[name][cold_slot], non_blocking=True)
    synchronize()
    for _, cold, temporary, _, cold_slot in items:
        for name in TENSORS:
            cold[name][cold_slot].copy_(temporary[name])


def swap_tensor_rows(hot, cold, temporary, hot_slot, cold_slot, synchronize):
    swap_tensor_rows_wave(((hot, cold, temporary, hot_slot, cold_slot),), synchronize)


def plan_waves(swaps, temp_slots):
    """Split an ordered plan into waves that may move concurrently.

    The policy may reuse a hot or cold slot later in the same plan; such a
    swap depends on the previous physical result and starts a new wave.
    Within a wave no (layer, slot) repeats, so the sequential and the wave
    execution leave identical weights and maps. Waves never exceed the TEMP
    row budget.
    """
    if temp_slots < 1:
        raise ValueError("Swap waves need at least one TEMP row")
    waves: list[list[Any]] = []
    wave: list[Any] = []
    used: set[tuple[int, str, int]] = set()
    for swap in swaps:
        keys = (
            (swap.layer, "hot", swap.hot_slot),
            (swap.layer, "cold", swap.cold_slot),
        )
        if wave and (len(wave) >= temp_slots or any(key in used for key in keys)):
            waves.append(wave)
            wave, used = [], set()
        wave.append(swap)
        used.update(keys)
    if wave:
        waves.append(wave)
    return waves


def temporary_row(temporary, index):
    return {name: tensor[index] for name, tensor in temporary.items()}


def _next_power_of_two(value):
    return 1 << max(int(value) - 1, 0).bit_length()


def mask_routes(ids, expert_map):
    """Routes whose expert is absent from `expert_map` become padding (-1)."""
    import torch

    num_experts = expert_map.shape[0]
    safe = ids.clamp(0, num_experts - 1).long()
    present = (ids >= 0) & (ids < num_experts) & (expert_map[safe] >= 0)
    return torch.where(present, ids, torch.full_like(ids, -1))


def physical_block_experts(logical_ids, post_padded, block, expert_map, num_experts):
    """Map per-block logical expert ids to physical rows on the device.

    Blocks at or beyond `post_padded` tokens are unused by the GEMM and may
    hold uninitialized ids; they become -1 without indexing anything.
    """
    import torch

    blocks = torch.arange(
        logical_ids.numel(), device=logical_ids.device, dtype=torch.int32
    )
    valid = (blocks * block) < post_padded.reshape(1)
    safe = torch.where(valid, logical_ids, torch.zeros_like(logical_ids))
    safe = safe.clamp(0, num_experts - 1)
    return torch.where(valid, expert_map[safe.long()], torch.full_like(logical_ids, -1))


def marlin_block_size(tokens, top_k, local_experts, global_experts, input_dtype):
    """The stock fused_marlin_moe M-block choice for one expert partition."""
    estimated = math.ceil(tokens * local_experts / global_experts)
    block = 8
    for block in (8, 16, 32, 48, 64):
        if estimated * top_k / local_experts / block < 0.9:
            break
    if input_dtype is not None and input_dtype.itemsize == 1:
        block = max(block, 16)
    return block


def replace_full_source_references(layer, method, hot, hot_kernel, hot_quant):
    """Remove raw-bank roots in both Parameters and quant/kernel objects.

    Reload is unsupported. Deliberately do not copy old Parameter attributes:
    those can contain loader state; the final hot tensors are already packed.
    """
    import torch

    for name in TENSORS:
        setattr(layer, name, torch.nn.Parameter(hot[name], requires_grad=False))
    method.moe_kernel = None if hot_kernel is NATIVE_KERNEL else hot_kernel
    method.moe_quant_config = hot_quant


class TierLayer:
    def __init__(
        self,
        index,
        name,
        layer,
        method,
        sources,
        slots,
        settings,
        pool=None,
        max_tokens=None,
    ):
        import torch

        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        self.index, self.name, self.layer, self.method = index, name, layer, method
        self.settings, self.device = settings, layer.w13_weight.device
        self.num_experts = layer.global_num_experts
        self.hot_slots, self.cold_slots = slots, self.num_experts - slots
        if not 0 < slots < self.num_experts:
            raise ValueError(
                "First tier version requires nonempty hot and cold partitions"
            )
        # Staging rows follow the hot rows in one bank so a single expert
        # map can address both; they hold transient copies of selected cold
        # experts during batch-1 decode and are outside the swap slot range.
        self.staging_slots = (
            method.moe.experts_per_token * settings.spec_rows if settings.staging else 0
        )
        # Spare rows for asynchronous exchanges follow the staging rows in
        # VRAM and the cold rows in RAM; logical slots map to physical rows
        # through hot_rows / cold_rows, which flip when a transfer commits.
        self.spare_slots = (
            settings.temp_slots if (settings.async_migration or settings.promote) else 0
        )
        staging_end = slots + self.staging_slots
        self.bank_rows = staging_end + self.spare_slots
        # RAM backing: the loader's pinned source (every expert, row = expert
        # id) is the RAM bank itself; nothing is copied or released on the
        # host, and there are no spare RAM rows because nothing is written.
        self.ram_backing = bool(settings.ram_backing)
        if self.ram_backing and not settings.promote:
            raise ValueError("RAM backing requires promote mode")
        # Global pool: the bank, staging rows, and tables are the pool's;
        # this layer's initial rows are a contiguous run of the pool.
        self.pool = pool
        if (pool is None) != (not settings.global_pool):
            raise ValueError("Global pool settings and pool object disagree")
        if pool is not None and not self.ram_backing:
            raise ValueError("Global pool requires RAM backing")
        self.native = settings.moe_kernel == "native"
        if self.native and not self.ram_backing:
            raise ValueError("Native backend requires RAM backing")
        self.max_tokens = max_tokens
        self.cold_rows_total = (
            self.num_experts if self.ram_backing else self.cold_slots + self.spare_slots
        )
        self.bank = {}
        self.hot = {}
        self.staging = {}
        self.cold_cpu = {}
        # Slicing without an independent allocation would retain the full bank.
        if pool is not None:
            self.spare_slots = 0
            self.bank_rows = pool.rows
            self.pool_offset = pool.offset(index)
            if self.pool_offset + slots > pool.tables.pool_rows:
                raise ValueError("Layer rows exceed the pool")
            for name, source in sources.items():
                bank = pool.bank[name]
                if source.shape[1:] != bank.shape[1:] or source.dtype != bank.dtype:
                    raise ValueError(f"{name}: layer rows differ from the pool rows")
        for name, source in sources.items():
            if pool is not None:
                self.bank[name] = pool.bank[name]
                start = self.pool_offset
                self.hot[name] = self.bank[name][start : start + slots]
                self.staging[name] = pool.staging[name]
            else:
                self.bank[name] = torch.zeros(
                    (self.bank_rows, *source.shape[1:]),
                    dtype=source.dtype,
                    device=self.device,
                )
                self.hot[name] = self.bank[name][:slots]
                self.staging[name] = self.bank[name][slots:staging_end]
            if self.ram_backing:
                self.cold_cpu[name] = source
            else:
                self.cold_cpu[name] = torch.empty(
                    (self.cold_rows_total, *source.shape[1:]),
                    dtype=source.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                self.cold_cpu[name][: self.cold_slots].copy_(source[slots:])
            self.hot[name].copy_(source[:slots], non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        self.cold = {
            name: get_accelerator_view_from_cpu_tensor(t)
            for name, t in self.cold_cpu.items()
        }
        self.hot_rows = list(
            range(self.pool_offset, self.pool_offset + slots)
            if pool is not None
            else range(slots)
        )
        self.cold_rows = list(
            range(slots, self.num_experts)
            if self.ram_backing
            else range(self.cold_slots)
        )
        self.vram_spares = SpareRing(range(staging_end, self.bank_rows))
        self.ram_spares = SpareRing(
            () if self.ram_backing else range(self.cold_slots, self.cold_rows_total)
        )
        self.hot_map_host = tuple(range(slots)) + (-1,) * self.cold_slots
        self.cold_map_host = (-1,) * slots + tuple(range(self.cold_slots))
        self.hot_map = self.cold_map = None
        self.promote_tables = self.promote_buffers = None
        self.promote_gate = False
        self.staging_rows = list(range(slots, staging_end))
        self.step_buffers = None
        if pool is not None:
            from .global_pool import allocate_step_buffers as pool_buffers

            self.staging_rows = pool.tables.staging_rows.tolist()
            # Scratch width is a power of two for the Triton program; the
            # staging capacity itself stays top_k.
            self.step_buffers = pool_buffers(
                self.device, self.num_experts, _next_power_of_two(self.staging_slots)
            )
            self.hot_map = pool.tables.layer_slice(pool.tables.hot_phys, index)
            self.cold_map = pool.tables.layer_slice(pool.tables.cold_phys, index)
        elif settings.promote:
            from .promote import allocate_step_buffers, allocate_tables

            # Device tables own the placement; the kernel maps alias them so
            # every path (prefill, verification) reads the current rows.
            self.promote_tables = allocate_tables(
                self.device,
                self.num_experts,
                slots,
                self.cold_slots,
                range(staging_end, self.bank_rows),
                ()
                if self.ram_backing
                else range(self.cold_slots, self.cold_rows_total),
                backing=self.ram_backing,
            )
            self.promote_buffers = allocate_step_buffers(
                self.device,
                self.num_experts,
                max(self.staging_slots, 1),
                self.staging_rows,
            )
            if settings.planner == "device" and self.device.type == "cuda":
                from . import device_lru

                # Planner state is allocated once, before any capture, and
                # kept on the tables; the gate starts closed.
                device_lru.allocate_state(self.promote_tables, self.staging_slots)
        self.publish_maps()
        # With spare rows a logical hot slot can live anywhere in the bank,
        # so the kernels address the whole bank / whole cold bank.
        if pool is not None or self.spare_slots:
            self.hot_tensors, self.hot_local = self.bank, self.bank_rows
            self.cold_local = self.cold_rows_total
        else:
            self.hot_tensors, self.hot_local = self.hot, slots
            self.cold_local = self.cold_slots
        self.native_workspaces = {}
        if self.native:
            # No Marlin kernel objects: the adapter is called with the bank
            # and a map. Workspaces are keyed by physical rows and shared by
            # every layer with that row count (layers run sequentially).
            from .native_nvfp4 import validate_bank

            for tensors in (self.hot_tensors, self.cold):
                validate_bank(tensors)
            self.hot_kernel: Any = NATIVE_KERNEL
            self.cold_kernel: Any = NATIVE_KERNEL
            self.bank_kernel: Any = NATIVE_KERNEL
            self.hot_quant = self.cold_quant = self.bank_quant = None
            self.native_workspace(self.hot_tensors)
            self.native_workspace(self.cold)
            if settings.native_prefill == "grouped":
                self.native_prefill_workspace(self.hot_tensors)
                self.native_prefill_workspace(self.cold)
        else:
            self.hot_kernel, self.hot_quant = self.make_kernel(
                self.hot_tensors, self.hot_local
            )
            self.cold_kernel, self.cold_quant = self.make_kernel(
                self.cold, self.cold_local
            )
            self.bank_kernel = self.bank_quant = None
            if self.staging_slots:
                if self.spare_slots or pool is not None:
                    self.bank_kernel, self.bank_quant = self.hot_kernel, self.hot_quant
                else:
                    self.bank_kernel, self.bank_quant = self.make_kernel(
                        self.bank, self.bank_rows
                    )
        self.marlin_workspace = None
        if settings.split == "fused" and not self.native:
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                marlin_make_workspace_new,
            )

            # One lock workspace per layer for the whole process, like the
            # dense Marlin linear path; captured graphs keep its address.
            self.marlin_workspace = marlin_make_workspace_new(self.device, 4)
        self.row_bytes = sum(t[0].numel() * t.element_size() for t in sources.values())
        self.hot_bytes = sum(t.numel() * t.element_size() for t in self.hot.values())
        # The pool's staging rows are charged once, by the pool.
        self.staging_bytes = (
            0 if pool is not None else self.row_bytes * self.staging_slots
        )
        self.spare_bytes = self.row_bytes * self.spare_slots
        self.cold_bytes = self.row_bytes * self.cold_slots
        # No physical RAM spare rows under backing: nothing is written.
        self.cold_spare_bytes = (
            0 if self.ram_backing else self.row_bytes * self.spare_slots
        )
        # Host bytes actually resident for this layer's RAM bank.
        self.host_bytes = self.row_bytes * self.cold_rows_total
        if self.hot_bytes + self.cold_bytes != self.row_bytes * self.num_experts:
            raise AssertionError("Exclusive partition lost or duplicated source bytes")
        self.coordinator = None

    def make_kernel(self, tensors, slots):
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
            make_nvfp4_moe_kernel,
        )

        proxy = SimpleNamespace(
            **tensors,
            w13_input_scale=None,
            w2_input_scale=None,
            swiglu_limit=getattr(self.layer, "swiglu_limit", None),
            swiglu_alpha=getattr(self.layer, "swiglu_alpha", None),
            swiglu_beta=getattr(self.layer, "swiglu_beta", None),
        )
        quant = self.method.get_fused_moe_quant_config(proxy)
        config = replace(self.method.moe, num_local_experts=slots)
        kernel = make_nvfp4_moe_kernel(
            quant,
            config,
            self.method.experts_cls,
            self.method.nvfp4_backend,
            routing_tables=None,
        )
        if kernel.prepare_finalize.supports_async():
            raise NotImplementedError(
                "Tier requires synchronous NoDPEP preparation/finalize"
            )
        _check_kernel_scales(kernel, tensors)
        return kernel, quant

    def publish_maps(self):
        import torch

        if getattr(self, "pool", None) is not None:
            # The kernel maps alias the pool tables from the start; the host
            # maps are refreshed from the device at snapshots only.
            return
        validate_partition(
            self.hot_map_host, self.cold_map_host, self.hot_slots, self.cold_slots
        )
        tables = getattr(self, "promote_tables", None)
        if tables is not None:
            # Promote mode: the kernel maps alias the device physical maps.
            # While the gate is closed (init, verification) the host maps are
            # authoritative and exchanges made on the host (verify's forced
            # swap) are pushed into the tables; once the gate is open the
            # device is the truth and the host only reads snapshots.
            if self.hot_map is None or self.cold_map is None:
                self.hot_map, self.cold_map = tables.hot_phys, tables.cold_phys
            if not getattr(self, "promote_gate", False):
                hot_rows: Any = self.hot_rows
                cold_rows: Any = self.cold_rows
                hot_logical = torch.tensor(self.hot_map_host, dtype=torch.int32)
                cold_logical = torch.tensor(self.cold_map_host, dtype=torch.int32)
                hot_physical = torch.tensor(
                    [hot_rows[v] if v >= 0 else -1 for v in self.hot_map_host],
                    dtype=torch.int32,
                )
                cold_physical = torch.tensor(
                    [cold_rows[v] if v >= 0 else -1 for v in self.cold_map_host],
                    dtype=torch.int32,
                )
                tables.hot_map.copy_(hot_logical)
                tables.cold_map.copy_(cold_logical)
                tables.hot_rows.copy_(torch.tensor(hot_rows, dtype=torch.int32))
                tables.cold_rows.copy_(torch.tensor(cold_rows, dtype=torch.int32))
                tables.hot_phys.copy_(hot_physical)
                tables.cold_phys.copy_(cold_physical)
                if not tables.backing:
                    shadow = tables.ram_shadow.tolist()
                    for expert, slot in enumerate(self.cold_map_host):
                        if slot >= 0:
                            shadow[cold_rows[slot]] = expert
                    tables.ram_shadow.copy_(torch.tensor(shadow, dtype=torch.int32))
            return
        # Pageable staging: the runtime finishes reading it before returning,
        # so no pinned host buffer can be overwritten during DMA. Device maps
        # hold physical rows; the host maps stay logical for the policy.
        hot_rows = getattr(self, "hot_rows", None)
        cold_rows = getattr(self, "cold_rows", None)
        hot = torch.tensor(
            [
                (hot_rows[v] if hot_rows is not None else v) if v >= 0 else -1
                for v in self.hot_map_host
            ],
            dtype=torch.int32,
        )
        cold = torch.tensor(
            [
                (cold_rows[v] if cold_rows is not None else v) if v >= 0 else -1
                for v in self.cold_map_host
            ],
            dtype=torch.int32,
        )
        if self.hot_map is None or self.cold_map is None:
            self.hot_map, self.cold_map = hot.to(self.device), cold.to(self.device)
            return
        # Captured graphs read these addresses: update them in place, on the
        # current stream, after the caller completed every previous use.
        self.hot_map.copy_(hot)
        self.cold_map.copy_(cold)

    def call(self, kernel, tensors, expert_map, x, weights, ids):
        return kernel.apply(
            x,
            tensors["w13_weight"],
            tensors["w2_weight"],
            weights,
            ids,
            activation=self.layer.activation,
            global_num_experts=self.num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=self.layer.apply_router_weight_on_input,
            # The outer vLLM runner owns shared expert execution/addition.
            shared_experts=None,
            shared_experts_input=None,
        )

    def split(self, x, weights, ids):
        staging = getattr(self, "staging_slots", 0)
        if staging and x.shape[0] * ids.shape[1] <= staging:
            if getattr(self, "pool", None) is not None:
                return self.split_global(x, weights, ids)
            if getattr(self, "promote_tables", None) is not None:
                return self.split_promote(x, weights, ids)
            return self.split_staged(x, weights, ids)
        if self.settings.split == "fused":
            return self.split_fused(x, weights, ids)
        # Marlin uses shared workspaces. Preserve the hot result before cold
        # runs, and keep the two calls on one stream. Neither prepare mutates x
        # on this verified BF16 NoDPEP path (router-on-input is rejected).
        hot = self.call(
            self.hot_kernel,
            getattr(self, "hot_tensors", self.hot),
            self.hot_map,
            x,
            weights,
            ids,
        ).clone()
        cold = self.call(self.cold_kernel, self.cold, self.cold_map, x, weights, ids)
        return hot.add_(cold)

    def split_global(self, x, weights, ids):
        """Batch-1 decode on the global pool: one step program, one copy
        launch from this layer's RAM bank, one chain through the step map."""
        from .global_pool import copy_in, step

        buffers = self.step_buffers
        if self.bank_kernel is None or buffers is None:
            raise RuntimeError("Global pool requires the bank kernel and buffers")
        step(self.pool.tables, self.index, ids, buffers)
        copy_in(self.cold, self.bank, buffers)
        return self._run_marlin_chains(
            x,
            weights,
            ids,
            (
                (
                    self.bank_kernel.fused_experts,
                    self.bank,
                    buffers.step_map,
                    self.bank_rows,
                ),
            ),
        )

    def split_promote(self, x, weights, ids):
        """Batch-1 decode in promote mode: plan, gather, evict, flip, one chain.

        Every step runs on the compute stream with fixed shapes: the planner
        (device LRU) decides promotions, victims, and staged-only misses; the
        flip updates the device tables and writes the copy lists and the
        step map; the copies move rows; the bank kernel runs through the
        step map. With the gate closed (startup, verification) every miss is
        staged only and the placement does not change.
        """
        from .promote import copy_rows, flip_step

        tables, buffers = self.promote_tables, self.promote_buffers
        if self.bank_kernel is None or tables is None or buffers is None:
            raise RuntimeError("Promote mode requires the bank kernel and tables")
        plan = self.plan_step(ids)
        flip_step(tables, plan, buffers, self.staging_rows)
        copy_rows(
            self.cold,
            self.bank,
            buffers.gather_src,
            buffers.gather_dst,
            buffers.gather_count,
        )
        if not tables.backing:
            copy_rows(
                self.bank,
                self.cold,
                buffers.evict_src,
                buffers.evict_dst,
                buffers.evict_count,
            )
        return self._run_marlin_chains(
            x,
            weights,
            ids,
            (
                (
                    self.bank_kernel.fused_experts,
                    self.bank,
                    buffers.step_map,
                    self.bank_rows,
                ),
            ),
        )

    def plan_step(self, ids):
        """Ask the configured planner for this step's promotions."""
        if self.settings.planner == "reference" or self.device.type != "cuda":
            from .promote import reference_plan

            return reference_plan(
                ids, self.promote_tables, self.staging_slots, self.promote_gate
            )
        from .device_lru import plan_step

        return plan_step(ids, self.promote_tables, self.staging_slots)

    def set_promote_gate(self, enabled):
        """Open or close promotion; closed leaves placement and recency alone."""
        self.promote_gate = bool(enabled)
        if getattr(self, "pool", None) is not None:
            from .global_pool import set_gate

            set_gate(self.pool.tables, enabled)
            return
        if self.promote_tables is None:
            return
        if self.settings.planner == "device" and self.device.type == "cuda":
            from . import device_lru

            (device_lru.open_gate if enabled else device_lru.close_gate)(
                self.promote_tables.lru_state
            )

    def pool_maps_host(self):
        hot_map, cold_map = cast(Any, self.hot_map), cast(Any, self.cold_map)
        return tuple(hot_map.tolist()), tuple(cold_map.tolist())

    def promote_snapshot(self):
        """Host copy of the device placement; validates and refreshes host maps."""
        from .promote import check_tables

        pool = getattr(self, "pool", None)
        if pool is not None:
            # Rows, not logical slots: the resident set of a layer floats.
            self.hot_map_host, self.cold_map_host = self.pool_maps_host()
            return
        tables = self.promote_tables
        if tables is None:
            return
        check_tables(tables, self.hot_slots, self.cold_slots)
        self.hot_map_host = tuple(tables.hot_map.tolist())
        self.cold_map_host = tuple(tables.cold_map.tolist())
        self.hot_rows = tables.hot_rows.tolist()
        self.cold_rows = tables.cold_rows.tolist()

    def split_staged(self, x, weights, ids):
        """Batch-1 decode: stage the selected cold experts, then one chain.

        `plan_staging` and `gather_staging` are fixed-shape and never touch
        the host, so this whole path captures into the decode graph. The
        bank kernel covers hot rows plus staging rows through the step's
        expert map; the cold RAM bank stays the owner and nothing is written
        back.
        """
        from .staging import gather_staging, plan_staging

        if self.bank_kernel is None:
            raise RuntimeError("Staging requires the bank kernel")
        # Staging rows start right after the logical hot region; spare rows
        # (if any) lie beyond them, so staged and spare rows never collide.
        gather, expert_map, count = plan_staging(
            ids, self.cold_map, self.hot_map, self.hot_slots, self.staging_slots
        )
        gather_staging(self.cold, self.staging, gather, count)
        return self._run_marlin_chains(
            x,
            weights,
            ids,
            (
                (
                    self.bank_kernel.fused_experts,
                    self.bank,
                    expert_map,
                    getattr(self, "bank_rows", self.hot_slots + self.staging_slots),
                ),
            ),
        )

    def prefill_scratch(self):
        """The shared device scratch bank for staged prefill (None when off)."""
        rows = self.settings.prefill_stage_rows
        if not rows:
            return None
        # Keyed by the bank signature too: layers with different row shapes
        # or dtypes must not share one scratch.
        signature = tuple(
            (name, tuple(self.cold[name].shape[1:]), str(self.cold[name].dtype))
            for name in TENSORS
        )
        key = (str(self.device), rows, signature)
        scratch = _PREFILL_SCRATCH.get(key)
        if scratch is None:
            import torch

            scratch = {
                name: torch.empty(
                    (rows, *self.cold[name].shape[1:]),
                    dtype=self.cold[name].dtype,
                    device=self.device,
                )
                for name in TENSORS
            }
            _PREFILL_SCRATCH[key] = scratch
        return scratch

    def stage_cold_partition(self, ids, scratch):
        """Copy this forward's routed cold experts into `scratch` and return
        the (staged, overflow) partitions replacing the UVA cold partition.

        Routed cold experts are taken in ascending id order up to the scratch
        rows; the rest keep the UVA partition through an overflow map. One
        host read of the routed count per layer (eager prefill only).
        """
        import torch

        from .promote import copy_rows

        cold_map = self.cold_map
        if cold_map is None:
            raise RuntimeError("Staged prefill needs the cold map")
        valid = (ids >= 0) & (ids < self.num_experts)
        safe = torch.where(valid, ids, torch.zeros_like(ids)).long()
        routed = valid & (cold_map[safe] >= 0)
        if _is_capturing(self.device):
            raise RuntimeError("Staged prefill cannot run during graph capture")
        # One dynamic-shape op (unique) per layer: unrouted slots map to a
        # sentinel past the last expert; one sentinel is always appended so
        # the sorted result always ends with it and no scalar is read back.
        sentinel = torch.full_like(safe, self.num_experts)
        marked = torch.cat(
            [torch.where(routed, safe, sentinel).reshape(-1), sentinel.reshape(-1)[:1]]
        )
        experts = torch.unique(marked)[:-1]
        rows = scratch[TENSORS[0]].shape[0]
        take = experts[:rows]
        count = int(take.numel())
        src_rows = cold_map[take].to(torch.int32)
        dst_rows = torch.arange(count, dtype=torch.int32, device=ids.device)
        copy_rows(
            self.cold,
            scratch,
            src_rows,
            dst_rows,
            torch.tensor([count], dtype=torch.int32, device=ids.device),
        )
        scratch_map = torch.full_like(cold_map, -1)
        scratch_map[take] = dst_rows.to(cold_map.dtype)
        overflow_map = cold_map.clone()
        overflow_map[take] = -1
        self.prefill_staged_rows = count
        self.prefill_overflow_rows = int(experts.numel()) - count
        kernel = self.cold_kernel.fused_experts
        staged = (kernel, scratch, scratch_map, count)
        overflow = (kernel, self.cold, overflow_map, self.prefill_overflow_rows)
        return staged, overflow

    def split_fused(self, x, weights, ids):
        """Both partitions into one per-slot row buffer, reduced once.

        Every (token, k) slot belongs to exactly one partition, so the hot and
        cold Marlin chains write disjoint rows of the same [tokens*k, hidden]
        buffer; padding slots belong to neither and stay at the zero fill.
        Compared with two modular kernel calls this removes both output
        allocations, both masked reductions, the hot clone, the add, and the
        prepare/finalize wrappers, while keeping one block alignment and two
        GEMMs per partition. Numerics are checked against the source kernel
        by init verification.
        """
        partitions: tuple[Any, ...] = (
            (
                self.hot_kernel.fused_experts,
                getattr(self, "hot_tensors", self.hot),
                self.hot_map,
                getattr(self, "hot_local", self.hot_slots),
            ),
            (
                self.cold_kernel.fused_experts,
                self.cold,
                self.cold_map,
                getattr(self, "cold_local", self.cold_slots),
            ),
        )
        scratch = self.prefill_scratch() if getattr(self, "native", False) else None
        if scratch is not None and x.shape[0] > self.settings.native_gemv_rows:
            staged, overflow = self.stage_cold_partition(ids, scratch)
            partitions = (partitions[0], staged, overflow)
        return self._run_marlin_chains(x, weights, ids, partitions)

    def native_workspace(self, tensors):
        """The shared adapter workspace for this bank's physical row count."""
        from .native_nvfp4 import allocate_workspace

        rows = tensors[TENSORS[0]].shape[0]
        key = (str(self.device), rows)
        workspace = _NATIVE_WORKSPACES.get(key)
        if workspace is None:
            if self.max_tokens is None:
                raise RuntimeError("Native workspaces need the runner token budget")
            workspace = allocate_workspace(
                tensors,
                self.max_tokens,
                self.method.moe.experts_per_token,
                num_experts=self.num_experts,
            )
            _NATIVE_WORKSPACES[key] = workspace
        self.native_workspaces[rows] = workspace
        return workspace

    def native_activation(self):
        from .native_loader import require_silu

        return require_silu(self.layer.activation)

    def native_prefill_workspace(self, tensors):
        """The grouped prefill kernel's workspace (its own allocator, with
        alignment scratch), one per physical row count, shared by layers."""
        from .native_prefill import allocate_workspace

        rows = tensors[TENSORS[0]].shape[0]
        key = (str(self.device), rows)
        workspace = _NATIVE_PREFILL_WORKSPACES.get(key)
        if workspace is None:
            if self.max_tokens is None:
                raise RuntimeError("Native workspaces need the runner token budget")
            workspace = allocate_workspace(
                tensors,
                self.max_tokens,
                self.method.moe.experts_per_token,
                num_experts=self.num_experts,
            )
            _NATIVE_PREFILL_WORKSPACES[key] = workspace
        return workspace

    def _run_native_chains(self, x, weights, ids, partitions):
        """One adapter call per partition; routes outside a partition are
        masked to padding so the adapter never records them as missing."""
        import torch

        from .native_nvfp4 import gemv

        compute: Any = gemv
        workspace_for = self.native_workspace
        if (
            x.shape[0] > self.settings.native_gemv_rows
            and self.settings.native_prefill == "grouped"
        ):
            from .native_prefill import prefill

            compute, workspace_for = prefill, self.native_prefill_workspace
        total = None
        for _experts, tensors, expert_map, _rows in partitions:
            if len(partitions) > 1:
                # Only valid routes owned by another partition become
                # padding; invalid ids reach the adapter unchanged so its
                # sticky error still records them. The map is indexed with
                # valid ids only.
                valid = (ids >= 0) & (ids < self.num_experts)
                safe = torch.where(valid, ids, torch.zeros_like(ids)).long()
                foreign = valid & (expert_map[safe] < 0)
                routed = torch.where(foreign, torch.full_like(ids, -1), ids)
            else:
                routed = ids
            out = compute(
                x,
                weights,
                routed.contiguous(),
                tensors,
                expert_map,
                workspace_for(tensors),
                activation=self.native_activation(),
            )
            if (
                len(partitions) == 1
                and x.shape[0] == 1
                and self.settings.native_output == "alias"
            ):
                # Consumed by the runner's out-of-place add (or the next
                # layer's combine) before the next gemv rewrites it.
                return out
            # The output aliases the workspace: own it before the next call.
            total = out.clone() if total is None else total.add_(out)
        return total

    def _run_marlin_chains(self, x, weights, ids, partitions):
        if getattr(self, "native", False):
            return self._run_native_chains(x, weights, ids, partitions)
        import torch

        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
            _fused_marlin_moe,
            marlin_moe_intermediate_size,
        )
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        from vllm.scalar_type import ScalarType
        from vllm.v1.worker.workspace import current_workspace_manager

        tokens, hidden = x.shape
        top_k = ids.shape[1]
        rows_count = tokens * top_k
        inner = marlin_moe_intermediate_size(
            self.hot["w13_weight"], self.hot["w2_weight"]
        )
        # Same manager as the stock kernels: stable addresses once locked.
        cache13, cache2, rows = current_workspace_manager().get_simultaneous(
            ((rows_count * max(2 * inner, hidden),), x.dtype),
            ((rows_count, inner), x.dtype),
            ((rows_count, hidden), x.dtype),
        )
        rows.zero_()
        for experts, tensors, expert_map, slots in partitions:
            block = marlin_block_size(
                tokens, top_k, slots, self.num_experts, experts.input_dtype
            )
            routed = ids
            if slots > self.num_experts:
                # Align by logical id (absent experts already padding), then
                # map the blocks to rows; the align op never sees a row id.
                routed = mask_routes(ids, expert_map)
                sorted_ids, logical_ids, post_padded = moe_align_block_size(
                    routed, block, self.num_experts, None, ignore_invalid_experts=True
                )
                expert_ids = physical_block_experts(
                    logical_ids, post_padded, block, expert_map, self.num_experts
                )
            else:
                sorted_ids, expert_ids, post_padded = moe_align_block_size(
                    ids,
                    block,
                    self.num_experts,
                    expert_map,
                    ignore_invalid_experts=True,
                )
            _fused_marlin_moe(
                hidden_states=x,
                w1=tensors["w13_weight"],
                w2=tensors["w2_weight"],
                bias1=experts.w1_bias,
                bias2=experts.w2_bias,
                w1_scale=experts.w1_scale,
                w2_scale=experts.w2_scale,
                topk_weights=weights,
                num_topk=top_k,
                quant_type=ScalarType.from_id(experts.quant_type_id),
                apply_router_weight_on_input=False,
                expert_map=expert_map,
                block_size_m=block,
                sorted_token_ids=sorted_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=post_padded,
                activation=self.layer.activation,
                activation_func=experts.activation,
                topk_ids=routed,
                input_global_scale1=experts.a1_gscale,
                input_global_scale2=experts.a2_gscale,
                global_scale1=experts.g1_alphas,
                global_scale2=experts.g2_alphas,
                g_idx1=experts.w13_g_idx,
                g_idx2=experts.w2_g_idx,
                sort_indices1=experts.w13_g_idx_sort_indices,
                sort_indices2=experts.w2_g_idx_sort_indices,
                w1_zeros=experts.w1_zp,
                w2_zeros=experts.w2_zp,
                workspace=self.marlin_workspace,
                intermediate_cache13=cache13,
                intermediate_cache2=cache2,
                output=rows,
                input_dtype=experts.input_dtype,
                is_k_full=experts.is_k_full,
                activation_config=experts.activation_config,
            )
        # Rows already carry the router weights (second GEMM multiplies them).
        return torch.sum(rows.view(tokens, top_k, hidden), dim=1)

    def stage_swap(self, old_expert, new_expert, hot_slot, cold_slot):
        """Validate against the current placement and advance the host maps.

        Physical rows move afterwards (`swap_tensor_rows_wave`) and the device
        maps are republished by the caller once the copies completed.
        """
        self.hot_map_host, self.cold_map_host = maps_after_swap(
            self.hot_map_host,
            self.cold_map_host,
            old_expert,
            new_expert,
            hot_slot,
            cold_slot,
        )

    def resolve_hot_row(self, hot_slot):
        rows = getattr(self, "hot_rows", None)
        return hot_slot if rows is None else rows[hot_slot]

    def resolve_cold_row(self, cold_slot):
        rows = getattr(self, "cold_rows", None)
        return cold_slot if rows is None else rows[cold_slot]

    def swap(self, old_expert, new_expert, hot_slot, cold_slot, temporary):
        """One sequential swap with its own waits; used by init verification."""
        if getattr(self, "pool", None) is not None:
            self.pool.host_swap(self.index, old_expert, new_expert)
            hot_map = cast(Any, self.hot_map)
            for name in TENSORS:
                row = int(hot_map[new_expert])
                self.bank[name][row].copy_(self.cold[name][new_expert])
            _current_stream(self.device).synchronize()
            self.hot_map_host, self.cold_map_host = self.pool_maps_host()
            return
        if getattr(self, "ram_backing", False):
            # The evicted expert's RAM row is intact: copy in, point the
            # logical cold slot at that row, nothing written to the host.
            hot_row, cold_row = self.resolve_hot_row(hot_slot), new_expert
            if self.resolve_cold_row(cold_slot) != cold_row:
                raise AssertionError("Backing cold slot must point at its expert row")
            self.stage_swap(old_expert, new_expert, hot_slot, cold_slot)
            for name in TENSORS:
                self.bank[name][hot_row].copy_(self.cold[name][cold_row])
            _current_stream(self.device).synchronize()
            self.cold_rows[cold_slot] = old_expert
            self.publish_maps()
            return
        self.stage_swap(old_expert, new_expert, hot_slot, cold_slot)
        swap_tensor_rows(
            getattr(self, "bank", self.hot),
            self.cold_cpu,
            temporary,
            self.resolve_hot_row(hot_slot),
            self.resolve_cold_row(cold_slot),
            _current_stream(self.device).synchronize,
        )
        self.publish_maps()

    def enqueue_swap(self, swap, stream):
        """Queue one exchange on the migration stream; nothing is published.

        The promoted cold expert is copied into a spare VRAM row and the
        evicted hot expert into a spare RAM row. The rows currently backing
        the two logical slots are only read, so the next forward may keep
        using the old placement. Each spare row's fence (the last reader of
        its previous content) is waited for before it is written.
        """
        from .async_migration import _on_stream, _stream_wait_event

        vram_spare = self.vram_spares.pop()
        ram_spare = self.ram_spares.pop()
        _stream_wait_event(stream, vram_spare.fence)
        _stream_wait_event(stream, ram_spare.fence)
        hot_row = self.resolve_hot_row(swap.hot_slot)
        cold_row = self.resolve_cold_row(swap.cold_slot)
        with _on_stream(stream):
            for name in TENSORS:
                self.bank[name][vram_spare.row].copy_(
                    self.cold[name][cold_row], non_blocking=True
                )
                self.cold_cpu[name][ram_spare.row].copy_(
                    self.bank[name][hot_row], non_blocking=True
                )
        return vram_spare, ram_spare

    def flip_swap(self, swap, vram_spare, ram_spare, retire_fence):
        """Point the logical slots at the transferred rows; retire the old.

        Called at a boundary after the transfer completed. The retired rows
        go back to their rings behind `retire_fence`, recorded on the
        compute stream after the last forward that read them.
        """
        old_row = self.hot_rows[swap.hot_slot]
        old_ram = self.cold_rows[swap.cold_slot]
        self.hot_rows[swap.hot_slot] = vram_spare.row
        self.cold_rows[swap.cold_slot] = ram_spare.row
        self.vram_spares.push(old_row, retire_fence)
        self.ram_spares.push(old_ram, retire_fence)
        self.stage_swap(swap.old_expert, swap.new_expert, swap.hot_slot, swap.cold_slot)

    def verify_initial(self, original_kernel, original, temporary):
        """Controlled nonzero BF16 routes exercise both partitions and a swap."""
        import torch

        k = self.method.moe.experts_per_token
        if min(self.hot_slots, self.cold_slots) < k:
            raise ValueError("Init verification requires top-k rows in each partition")
        width = self.method.moe.hidden_dim
        # Deterministic, bounded, finite input with RMS near one; no RNG state
        # mutation and no dependency on model-runner synthetic zero inputs.
        x = torch.sin(
            torch.arange(4 * width, device=self.device, dtype=torch.float32) * 0.013
        ).reshape(4, width)
        x = x.to(torch.bfloat16)
        hot_ids = list(range(k))
        cold_ids = list(range(self.hot_slots, self.hot_slots + k))
        mixed = [hot_ids[j] if j % 2 == 0 else cold_ids[j] for j in range(k)]
        ids = torch.tensor(
            [hot_ids, cold_ids, mixed, list(reversed(mixed))],
            dtype=torch.int32,
            device=self.device,
        )
        weights = torch.arange(1, k + 1, dtype=torch.float32, device=self.device)
        weights = (weights / weights.sum()).expand(4, -1).contiguous()
        if getattr(self, "native", False):
            from .native_nvfp4 import gemv

            # The adapter over the full checkpoint banks (every expert's own
            # row) is the reference; the tier must reproduce it exactly.
            source = {name: original[name].data for name in TENSORS}
            identity = torch.arange(
                self.num_experts, dtype=torch.int32, device=self.device
            )
            reference = gemv(
                x.clone(),
                weights.clone(),
                ids.clone(),
                source,
                identity,
                self.native_workspace(source),
                activation=self.native_activation(),
            ).clone()
        else:
            reference = self.call(
                original_kernel, original, None, x.clone(), weights.clone(), ids.clone()
            ).clone()
        metrics = []
        padding_checks = []
        for stage in ("initial", "forced_swap", "restored"):
            if stage == "forced_swap":
                self.swap(0, self.hot_slots, 0, 0, temporary)
            elif stage == "restored":
                self.swap(self.hot_slots, 0, 0, 0, temporary)
            actual = self.split(x.clone(), weights.clone(), ids.clone()).clone()
            metrics.append(compare_outputs(actual, reference, self.name, stage))
            pad_ids = torch.full((1, k), -1, dtype=ids.dtype, device=self.device)
            pad = self.split(x[:1].clone(), weights[:1].clone(), pad_ids).clone()
            torch.testing.assert_close(
                pad, torch.zeros_like(pad), rtol=0, atol=0, equal_nan=False
            )
            mixed_ids = torch.cat((ids, pad_ids))
            mixed_output = self.split(
                torch.cat((x, x[:1])), torch.cat((weights, weights[:1])), mixed_ids
            ).clone()
            compare_outputs(
                mixed_output[:4], reference, self.name, stage + "_mixed_padding"
            )
            torch.testing.assert_close(
                mixed_output[4:],
                torch.zeros_like(mixed_output[4:]),
                rtol=0,
                atol=0,
                equal_nan=False,
            )
            padding_checks.append(
                {
                    "stage": stage,
                    "all_padding_exact_zero": True,
                    "mixed_padding_exact_zero": True,
                }
            )
        staged_checks = []
        self.set_promote_gate(False)
        if self.staging_slots:
            # Batch-1 rows take the staged path: hot only, cold only, mixed,
            # and padding, each against the same source-kernel reference.
            for row, label in enumerate(("hot", "cold", "mixed", "mixed_reversed")):
                single = self.split(
                    x[row : row + 1].clone(),
                    weights[row : row + 1].clone(),
                    ids[row : row + 1].clone(),
                ).clone()
                staged_checks.append(
                    compare_outputs(
                        single, reference[row : row + 1], self.name, "staged_" + label
                    )
                )
            pad_ids = torch.full((1, k), -1, dtype=ids.dtype, device=self.device)
            pad = self.split(x[:1].clone(), weights[:1].clone(), pad_ids).clone()
            torch.testing.assert_close(
                pad, torch.zeros_like(pad), rtol=0, atol=0, equal_nan=False
            )
            staged_checks.append({"stage": "staged_padding", "exact_zero": True})
        torch.cuda.current_stream(self.device).synchronize()
        LOGGER.warning(
            "LAB_EXPERT_TIER_VERIFY_INIT %s",
            json.dumps(
                {
                    "layer": self.name,
                    "layer_index": self.index,
                    "passed": True,
                    "stages": metrics,
                    "staged_checks": staged_checks,
                    "input_nonzero": int(torch.count_nonzero(x).item()),
                    "input_tokens": 4,
                    "shared_experts_reexecuted": False,
                    "padding_checks": padding_checks,
                    "scope": "controlled_BF16_source_vs_hot_plus_cold_and_forced_swap",
                },
                sort_keys=True,
            ),
        )

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        coordinator = self.coordinator
        if coordinator is None or layer is not self.layer:
            raise RuntimeError("Uninitialized or incorrectly attached expert tier")
        with coordinator.lock:
            try:
                coordinator.begin_layer(self, x, topk_weights, topk_ids)
                result = self.split(x, topk_weights, topk_ids)
                coordinator.end_layer(self)
                return result
            except Exception:
                coordinator.poisoned = True
                raise


def compare_outputs(actual, reference, name, stage):
    import torch

    a, r = actual.float(), reference.float()
    finite = bool(torch.isfinite(a).all().item() and torch.isfinite(r).all().item())
    delta = (a - r).abs()
    metrics = {
        "stage": stage,
        "finite": finite,
        "rtol": VERIFY_RTOL,
        "atol": VERIFY_ATOL,
        "max_abs": delta.max().item() if finite else None,
        "max_rel": (delta / r.abs().clamp_min(1e-6)).max().item() if finite else None,
    }
    try:
        if not finite:
            raise AssertionError("Nonfinite routed output")
        torch.testing.assert_close(
            actual,
            reference,
            rtol=VERIFY_RTOL,
            atol=VERIFY_ATOL,
            equal_nan=False,
            check_dtype=True,
            check_device=True,
        )
    except AssertionError:
        LOGGER.error(
            "LAB_EXPERT_TIER_VERIFY_INIT %s",
            json.dumps({"layer": name, "passed": False, **metrics}, sort_keys=True),
        )
        raise
    return metrics


def _current_stream(device):
    import torch

    if device.type != "cuda":
        return SimpleNamespace(cuda_stream=0, synchronize=lambda: None)
    return torch.cuda.current_stream(device)


def _is_capturing(device):
    import torch

    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


@dataclass(frozen=True)
class LegacyRoutes:
    """Host routing lists for one forward; the coordinator observes them."""

    routes: Any
    activity: Any
    mask: Any
    tokens: int


@dataclass(frozen=True)
class DeviceSnapshot:
    """Heat already advanced on the device; the coordinator only plans.

    `heat` is a host float64 [layers, experts] array that the policy imports
    verbatim (integer counts were accumulated per forward, decayed, then
    added once, in the policy's order). `tokens`, `forwards`, `route_hot`,
    and `route_total` are cumulative device totals; the coordinator turns
    them into deltas against the previous snapshot it consumed and derives
    the token delta from the policy's own total. Observers must not advance
    device state while heat is disabled, so a snapshot may only arrive with
    heat enabled.
    """

    heat: Any
    tokens: int
    forwards: int
    route_hot: int
    route_total: int
    verified: bool = True


@dataclass(frozen=True)
class Deferred:
    """No host readback this forward; `flush` delivers the snapshot later."""

    forwards: int = 1


class RecordObserver:
    """Default observer: static device records, one D2H per forward.

    Contract shared with device-side observers:
    - `record_layer` runs inside the (possibly captured) forward and may only
      touch fixed-address device state.
    - `finish` runs on the runner's stream at the model boundary and returns
      LegacyRoutes, DeviceSnapshot, or Deferred. It must raise on an invalid
      device record, which poisons the coordinator.
    - `flush` returns a pending snapshot or None; used at stats, forced
      resync, and shutdown.
    """

    def __init__(self, **_config):
        # Device observers take the policy configuration; this one needs none.
        self.records: Any = None
        self.records_host: Any = None
        self.device: Any = None

    @property
    def capacity(self):
        return 0 if self.records is None else self.records.shape[0]

    def allocate(self, device, layers, top_k, max_tokens, num_experts=None):
        import torch

        if self.records is not None:
            raise RuntimeError("Tier routing records are already allocated")
        if top_k < 1 or max_tokens < 1:
            raise ValueError("Routing records need positive top-k and token capacity")
        # Token-major so a forward's rows are one contiguous prefix.
        shape = (max_tokens, layers, 2 * top_k + 1)
        self.device = device
        self.records = torch.zeros(shape, dtype=torch.int32, device=device)
        self.records_host = torch.zeros(
            shape, dtype=torch.int32, device="cpu", pin_memory=device.type == "cuda"
        )

    # This observer reads routing back and counts hot hits on the host.
    reports_route_hot = True

    def record_layer(self, layer_index, rows, ids, active, valid, hot_map=None):
        """Record one layer; `hot_map` is the layer's current device map.

        Device observers count hot hits from it (`hot_map[id] >= 0` over
        valid, active, in-range lanes, duplicates counted per selection);
        it is updated in place after exchanges, so it reflects the placement
        this forward ran on. Unused here: the host readback has the maps.
        """
        import torch

        packed = torch.cat(
            (
                ids.to(torch.int32),
                active.to(torch.int32),
                valid[:, None].to(torch.int32),
            ),
            dim=1,
        )
        if packed.shape[1] != self.records.shape[2]:
            raise ValueError("Routing record width does not match the allocation")
        # Graph-safe: a fixed destination written on the forward's stream.
        self.records[:rows, layer_index].copy_(packed)

    def finish(
        self, rows, valid_rows, heat_enabled, stream, num_experts, is_decode=None
    ):
        # One batched D2H at the model boundary, matching update_from_graph.
        # `is_decode` only steers device snapshot timing; legacy records
        # observe every forward the same way.
        # This also completes all hot/cold uses before any RAM TEMP migration.
        self.records_host[:rows].copy_(self.records[:rows], non_blocking=True)
        stream.synchronize()
        packed = self.records_host[:rows].transpose(0, 1).tolist()
        return LegacyRoutes(*unpack_routes(packed, num_experts))

    def flush(self):
        return None

    def on_heat_enabled(self):
        """Open a persistent device gate; Python flags never reach a replay."""
        return None


# name -> class or "module:Class"; device observers register here.
OBSERVERS: dict[str, Any] = {
    "records": RecordObserver,
    "device": "vllm._lab_expert_tier.heat_device:DeviceObserver",
}


def make_observer(name, **config):
    """Instantiate a registered observer with the policy configuration."""
    import importlib

    target = OBSERVERS.get(name)
    if target is None:
        raise ValueError(f"Unknown expert tier observer {name!r}")
    if isinstance(target, str):
        module_name, _, class_name = target.partition(":")
        target = getattr(importlib.import_module(module_name), class_name)
    return target(**config)


def _is_snapshot(result):
    return hasattr(result, "heat")


def _is_deferred(result):
    return (
        hasattr(result, "forwards")
        and not hasattr(result, "heat")
        and not hasattr(result, "routes")
    )


class TierCoordinator:
    def __init__(self, layers, settings, temporary, observer=None):
        self.layers, self.settings, self.temporary = layers, settings, temporary
        self.pool = None
        self._control_rejected = None
        hot_slots = tuple(layer.hot_slots for layer in layers)
        self.policy = TierPolicy(
            len(layers),
            layers[0].num_experts,
            # The policy takes one count while every layer agrees; a
            # per-layer tuple is the policy-side extension for explicit
            # allocations.
            cast(Any, hot_slots[0] if len(set(hot_slots)) == 1 else hot_slots),
            **settings.policy_kwargs(),
        )
        self.lock = threading.Lock()
        self.poisoned, self.stream_id = False, None
        self.stream: Any = None
        # The observer owns every per-forward device record. The default
        # keeps static routing records and reads them back once per forward.
        self.observer = RecordObserver() if observer is None else observer
        if settings.promote:
            observation_only = getattr(self.observer, "set_observation_only", None)
            if observation_only is not None:
                observation_only(True)
        self.device: Any = None
        self.recorded = 0  # layers recorded by the forward in progress
        self.forward_rows: int | None = None  # recorded, not yet finished
        # Cumulative totals of the last consumed device snapshot, per session.
        self.snapshot_totals: dict[str, int] = {}
        self.snapshot_session: Any = None
        # Whether the last device snapshot counted hot hits from real maps.
        self.route_hot_measured = False
        # One asynchronous exchange plan in flight, at most.
        self.pending: Any = None
        self.stats = cast("dict[str, int | float]", Counter())
        self._verify_version = None
        self.draft_reserve: dict[str, Any] | None = None
        self.draft_resident: dict[str, Any] | None = None
        self.per_layer_swaps = [0] * len(layers)
        # MRv2 builtin kernel warmup uses fake requests with mask=False.
        # Only the successful compile_or_warm_up_model tail enables heat.
        self.heat_enabled = False

    @property
    def records(self):
        return getattr(self.observer, "records", None)

    @property
    def records_host(self):
        return getattr(self.observer, "records_host", None)

    def allocate_records(self, device, top_k, max_tokens):
        self.device = device
        self.observer.allocate(
            device,
            len(self.layers),
            top_k,
            max_tokens,
            num_experts=self.layers[0].num_experts,
        )

    def adopt_stream(self, stream, boundary):
        if self.stream_id == stream.cuda_stream:
            return  # The common path adds no event, wait, or synchronization.
        if self.stream_id is not None:
            if not boundary or self.recorded:
                raise NotImplementedError(
                    "Tier cannot change CUDA stream within a model forward"
                )
            # Warmup, graph capture, and real execution use different
            # streams. At a complete-model boundary, finish previous
            # workspace/weight/map uses before handing ownership to the new
            # stream. Keep the old stream alive and do not publish the new
            # one if this wait fails.
            self.stream.synchronize()
        self.stream, self.stream_id = stream, stream.cuda_stream

    def discard_unconsumed(self, reason):
        # Startup warmup and graph capture call the model directly, so they
        # legitimately leave a recorded forward that no runner hook finishes.
        # After heat is enabled every forward must be finished, or heat and
        # migration would silently skip real tokens.
        if self.heat_enabled:
            raise RuntimeError(f"Recorded tier forward was never finished: {reason}")
        self.stats["dropped_startup_records"] += 1
        self.forward_rows = None

    def begin_layer(self, tier, x, weights, ids):
        import torch

        from vllm.forward_context import get_forward_context

        if self.poisoned:
            raise RuntimeError("Expert tier is poisoned by a previous failure")
        if self.device is None or not self.observer.capacity:
            raise RuntimeError("Tier routing records are not allocated")
        if tier.index != self.recorded:
            raise RuntimeError("Tier requires one sequential full-model forward")
        rows = x.shape[0]
        stream = _current_stream(tier.device)
        if tier.index == 0:
            if self.forward_rows is not None:
                self.discard_unconsumed("next forward started")
            if not 0 < rows <= self.observer.capacity:
                raise ValueError("Forward exceeds the tier routing record capacity")
            if (
                self.stream_id is not None
                and self.stream_id != stream.cuda_stream
                and _is_capturing(tier.device)
            ):
                # Adopting waits on the previous stream, which is illegal
                # inside a capture. vLLM warms up on the capture stream
                # first, so this only guards a changed capture protocol.
                raise RuntimeError("Tier cannot adopt a stream during graph capture")
            self.forward_rows = rows
        elif rows != self.forward_rows:
            raise ValueError("Token count changed within a model forward")
        self.adopt_stream(stream, tier.index == 0)
        if (
            x.ndim != 2
            or ids.ndim != 2
            or weights.shape != ids.shape
            or ids.shape[0] != x.shape[0]
            or x.device != tier.device
            or ids.device != tier.device
            or weights.device != tier.device
            or ids.dtype not in (torch.int32, torch.int64)
            or x.dtype != torch.bfloat16
            or ids.shape[1] != tier.method.moe.experts_per_token
        ):
            raise ValueError("Unexpected tier routing/input shape, dtype, or device")
        mask = get_forward_context().is_padding
        if mask is None:
            raise NotImplementedError("Tier requires MRv2's explicit padding mask")
        if (
            mask.dtype != torch.bool
            or mask.ndim != 1
            or mask.shape[0] != x.shape[0]
            or mask.device != x.device
        ):
            raise ValueError("Expected one boolean padding flag per routing row")
        if self.settings.record_kernel and self._fused_record(
            tier, rows, ids, weights, mask
        ):
            self.recorded += 1
            return
        valid = ~mask
        allowed = ((ids >= 0) & (ids < tier.num_experts)) | (
            (ids == -1) & mask[:, None]
        )
        valid_weights = (torch.isfinite(weights) & (weights >= 0)) | mask[:, None]
        # Device assertion avoids 48 per-layer routing D2H synchronizations.
        # Invalid real IDs can never silently mask out a selected expert.
        torch._assert_async(
            (allowed & valid_weights).all(),
            "Invalid routing: -1 requires padding; "
            "real weights must be finite/nonnegative",
        )
        self.observer.record_layer(
            tier.index,
            rows,
            ids,
            weights != 0,
            valid,
            hot_map=getattr(tier, "hot_map", None),
        )
        self.recorded += 1

    def _fused_record(self, tier, rows, ids, weights, mask):
        """One-launch record through the device observer's tensors.

        Returns False (caller takes the old path) when the observer has no
        kernel targets or the rows exceed the fused width.
        """
        from .device_record import MAX_LANES, record

        targets = getattr(self.observer, "kernel_targets", None)
        if targets is None or rows * ids.shape[1] > MAX_LANES:
            return False
        hot_map = getattr(tier, "hot_map", None)
        record(targets(), tier.index, rows, ids, weights, mask, hot_map)
        cast(Any, self.observer).note_kernel_record(tier.index, rows, hot_map)
        return True

    def end_layer(self, tier):
        if tier.index + 1 != len(self.layers):
            return
        self.recorded = 0
        if _is_capturing(tier.device):
            # Capture-time rows are dummy padding and nobody finishes them.
            self.stats["captured_forwards"] += 1
            self.forward_rows = None

    def finish_forward(self, rows, valid_rows=None, is_decode=None):
        """Runner-side model boundary: the one host copy per forward.

        Called after eager forwards and CUDA Graph replays alike. A replay
        executes no Python in the layers, so the runner supplies the padded
        row count and the static records carry this forward's routing/mask.
        `valid_rows` is the runner's real token count (None when unknown); it
        is an upper bound for observers that defer host readback, never a
        substitute for the device padding mask. `is_decode` is the runner's
        statement that this forward held no prefill (None when unknown):
        device observers may snapshot a multi-row decode (verify) step, but
        never a small prefill.
        """
        with self.lock:
            if self.poisoned:
                raise RuntimeError("Expert tier is poisoned by a previous failure")
            try:
                self._finish_forward(rows, valid_rows, is_decode)
            except Exception:
                self.poisoned = True
                raise

    def flush(self, plan=False):
        """Collect a deferred device observation outside a forward boundary.

        Observation only by default (stats and shutdown); `plan=True` is the
        forced-resync entry and may migrate.
        """
        with self.lock:
            if self.poisoned:
                raise RuntimeError("Expert tier is poisoned by a previous failure")
            try:
                self._flush_locked(plan)
            except Exception:
                self.poisoned = True
                raise

    def _flush_locked(self, plan):
        # A started transaction is decided before any import, whatever
        # `plan` says; `plan` only controls whether a new plan may be made.
        self.settle_pending(wait=True)
        result = self.observer.flush()
        if result is not None:
            self._consume(result, plan)

    def poll_verify_file(self):
        """Run a pool verification when the trigger file's mtime changed
        (checked every 16 forwards; the check itself is one stat call)."""
        path = getattr(self.settings, "verify_file", "")
        if not path or self.stats["model_forwards"] % 16:
            return None
        import os

        try:
            version = os.stat(path).st_mtime_ns
        except OSError:
            return None
        if version == self._verify_version:
            return None
        self._verify_version = version
        return self.verify_pool()

    def verify_pool(self):
        """Compare every resident expert row of every layer's device bank with
        the host source row (RAM backing: source row = expert id), bytewise,
        on the current stream; log and return the per-layer mismatch counts.
        Runs at a forward boundary only, never inside a capture."""
        import torch

        if _is_capturing(self.device):
            raise RuntimeError("Pool verification cannot run during graph capture")
        report: dict[str, Any] = {
            "forwards": int(self.stats["model_forwards"]),
            "layers": {},
        }
        total = checked = 0
        for tier in self.layers:
            hot_map = tier.hot_map
            if hot_map is None:
                continue
            experts = torch.nonzero(hot_map >= 0).flatten()
            if experts.numel() == 0:
                continue
            rows = hot_map[experts].long()
            bank = getattr(tier, "hot_tensors", None)
            if bank is None:
                bank = tier.hot
            bad = torch.zeros(experts.numel(), dtype=torch.bool, device=experts.device)
            for name in TENSORS:
                # Read-only: index_select copies; nothing in the bank or the
                # maps is touched, and no mismatch is repaired here.
                device_rows = bank[name].index_select(0, rows)
                source_rows = tier.cold[name].index_select(0, experts.long())
                bad |= (device_rows != source_rows).flatten(1).any(1)
            count = int(bad.sum())
            checked += int(experts.numel())
            total += count
            if count:
                report["layers"][tier.index] = {
                    "mismatched": count,
                    "experts": experts[bad][:8].tolist(),
                    "rows": rows[bad][:8].tolist(),
                }
        report["checked_rows"] = checked
        report["mismatched_rows"] = total
        LOGGER.warning("LAB_EXPERT_TIER_VERIFY_POOL %s", json.dumps(report))
        self.stats["verify_pool_runs"] += 1
        self.stats["verify_pool_mismatched_rows"] += total
        return report

    def _finish_forward(self, rows, valid_rows=None, is_decode=None):
        if valid_rows is not None and not 0 <= valid_rows <= rows:
            raise ValueError("Runner valid token count exceeds the padded rows")
        if self.device is None or not self.observer.capacity:
            raise RuntimeError("Tier routing records are not allocated")
        if self.recorded:
            raise RuntimeError("Model forward finished with incomplete tier layers")
        if _is_capturing(self.device):
            raise RuntimeError("Tier forward cannot be finished during graph capture")
        if not 0 < rows <= self.observer.capacity:
            raise ValueError("Finished forward exceeds the routing record capacity")
        if self.forward_rows is None:
            self.stats["replayed_forwards"] += 1
        else:
            if self.forward_rows != rows:
                raise ValueError(
                    "Runner token count disagrees with the recorded forward"
                )
            self.forward_rows = None
            self.stats["recorded_forwards"] += 1
        stream = _current_stream(self.device)
        self.adopt_stream(stream, True)
        # A pending exchange decides here, before this forward is observed:
        # the flip and the policy commit precede any new observation or plan.
        self.settle_pending(wait=True)
        result = self.observer.finish(
            rows,
            valid_rows,
            self.heat_enabled,
            stream,
            self.layers[0].num_experts,
            is_decode=is_decode,
        )
        # One real forward, counted exactly once whatever the observer returns.
        self.stats["model_forwards"] += 1
        if not self.heat_enabled:
            self.stats["ignored_startup_forwards"] += 1
        self._consume(result, plan=True)
        self.poll_verify_file()
        if self.stats["model_forwards"] % self.settings.stats_every == 0:
            self.report()

    def _consume(self, result, plan):
        """Apply one observer result. Forward counters are not touched here.

        `plan` allows planning and migration; a flush at exit or before a
        report collects the observation only, so no GPU work starts outside
        a forward boundary.
        """
        # Device observers return their own result types without importing
        # this module, so results are recognized by shape, not by class.
        if _is_snapshot(result):
            if not self.heat_enabled:
                raise RuntimeError(
                    "Device observers must not advance heat before it is enabled"
                )
            if not getattr(result, "verified", True) or getattr(result, "error", False):
                raise RuntimeError(
                    "Device routing validation failed (delayed device check)"
                )
            importer = getattr(self.policy, "import_snapshot", None)
            if importer is None:
                raise NotImplementedError("Policy cannot import device snapshots")
            tokens_before = self.policy.tokens_total
            started = time.perf_counter()
            importer(result)
            acknowledge = getattr(self.observer, "acknowledge_snapshot", None)
            if acknowledge is not None:
                acknowledge(result)
            self.stats["policy_observe_seconds"] += time.perf_counter() - started
            # Snapshot totals are cumulative per device session; account the
            # increments only, and tokens from the policy's own total.
            session = getattr(result, "session_id", None)
            if session != self.snapshot_session:
                self.snapshot_session, self.snapshot_totals = session, {}
            for key in ("forwards", "route_hot", "route_total"):
                total = getattr(result, key, 0)
                delta = total - self.snapshot_totals.get(key, 0)
                if delta < 0:
                    raise RuntimeError(f"Device snapshot {key} went backwards")
                self.snapshot_totals[key] = total
                stat = "snapshot_forwards" if key == "forwards" else key
                self.stats[stat] += delta
            self.stats["model_tokens"] += self.policy.tokens_total - tokens_before
            self.stats["device_snapshots"] += 1
            self.route_hot_measured = bool(
                getattr(result, "route_hot_available", False)
            )
            if plan:
                self._plan_and_migrate()
            # The device restarts from the policy state that now holds,
            # whether or not a plan was committed.
            rebase = getattr(self.observer, "rebase", None)
            if rebase is not None:
                rebase(
                    tokens_total=self.policy.tokens_total,
                    version=self.policy.version,
                    last_sync_tokens=self.policy.last_sync_tokens,
                )
            return
        if _is_deferred(result):
            # The observation arrives with a later snapshot; nothing may be
            # planned against stale heat.
            self.stats["deferred_forwards"] += getattr(result, "forwards", 1)
            return
        if not hasattr(result, "routes"):
            raise TypeError("Observer returned an unknown result type")
        routes, activity, mask, tokens = (
            result.routes,
            result.activity,
            result.mask,
            result.tokens,
        )
        if not self.heat_enabled:
            return
        if tokens:
            # Measure the routing placement used by this forward, before heat
            # observation or migration changes it. CPU records already exist.
            for layer, route_rows, active_rows in zip(self.layers, routes, activity):
                hot_map = layer.hot_map_host
                hot, total = 0, 0
                for valid, ids, active in zip(mask, route_rows, active_rows):
                    if valid:
                        for expert, selected in zip(ids, active):
                            if selected:
                                total += 1
                                hot += hot_map[expert] >= 0
                self.stats["route_hot"] += hot
                self.stats["route_total"] += total
            started = time.perf_counter()
            self.policy.observe_step(
                routes, tokens, routing_weights_by_layer=activity, valid_token_mask=mask
            )
            self.stats["policy_observe_seconds"] += time.perf_counter() - started
            self.stats["model_tokens"] += tokens
            if plan:
                self._plan_and_migrate()
        else:
            self.stats["ignored_synthetic_forwards"] += 1

    def _plan_and_migrate(self):
        if self.settings.promote:
            # The device LRU owns the placement; the heat policy observes only.
            self.stats["promote_steps_observed"] += 1
            return
        if True:
            started = time.perf_counter()
            plan = self.policy.plan_resync()
            self.stats["policy_plan_seconds"] += time.perf_counter() - started
            if plan is not None:
                if plan.swaps and self.settings.async_migration:
                    from .async_migration import preflight

                    if self.pending is not None:
                        raise RuntimeError("An exchange plan is already pending")
                    budgets = {
                        i: min(layer.vram_spares.free, layer.ram_spares.free)
                        for i, layer in enumerate(self.layers)
                    }
                    verdict = preflight(plan.swaps, budgets)
                    if verdict.eligible:
                        self._begin_async(plan, verdict)
                        return
                    self.stats["sync_fallbacks"] += 1
                    self.stats["fallback_" + str(verdict.reason)] += 1
                migration_started = time.perf_counter() if plan.swaps else None
                self.migrate(plan.swaps)
                if plan.swaps:
                    # RAM TEMP already waits for its two DMA phases. Include
                    # the final in-place map publication too: this is
                    # completed wall time, not merely enqueue time, and it
                    # guarantees the next forward (or replay) sees the new
                    # weights and maps.
                    if self.stream is None:
                        raise RuntimeError(
                            "Migration requires an adopted execution stream"
                        )
                    self.stream.synchronize()
                    self.stats["migration_wall_seconds"] += time.perf_counter() - cast(
                        float, migration_started
                    )
                # Copy errors poison the engine before policy publication.
                started = time.perf_counter()
                self.policy.commit(plan)
                self.stats["resyncs"] += 1
                for i, layer in enumerate(self.layers):
                    if (
                        tuple(self.policy.expert_to_hot[i]) != layer.hot_map_host
                        or tuple(self.policy.expert_to_cold[i]) != layer.cold_map_host
                    ):
                        raise AssertionError("Policy and physical tier maps diverged")
                self.stats["policy_commit_seconds"] += time.perf_counter() - started

    def _begin_async(self, plan, verdict):
        """Queue a whole eligible plan on the migration stream; commit later."""
        from .async_migration import (
            MigrationTransaction,
            _migration_stream,
            _record_event,
            _stream_wait_event,
        )

        started = time.perf_counter()
        stream = _migration_stream(self.device)
        # Everything this forward queued on the compute stream finishes
        # before the migration stream reads the old rows.
        _stream_wait_event(stream, _record_event(_current_stream(self.device)))
        transaction = MigrationTransaction(plan, verdict, enqueued_at=started)
        # Tracked from the first copy: a failure mid-enqueue leaves the
        # transaction visible (and the tier poisoned), never half-forgotten.
        self.pending = transaction
        for swap in plan.swaps:
            layer = self.layers[swap.layer]
            vram_spare, ram_spare = layer.enqueue_swap(swap, stream)
            transaction.entries.append((swap.layer, swap, vram_spare, ram_spare))
        transaction.transfer_event = _record_event(stream)
        self.stats["async_plans"] += 1
        self.stats["async_swaps_enqueued"] += len(plan.swaps)
        self.stats["async_enqueue_seconds"] += time.perf_counter() - started

    def settle_pending(self, wait):
        """Flip and commit the pending plan once its transfers completed.

        Returns True when a plan was committed. With `wait=False` an
        incomplete transfer leaves the old placement in force.
        """
        from .async_migration import _record_event, _stream_wait_event

        transaction = self.pending
        if transaction is None:
            return False
        event = transaction.transfer_event
        if not event.query():
            if not wait:
                self.stats["async_pending_boundaries"] += 1
                return False
            started = time.perf_counter()
            event.synchronize()
            self.stats["async_wait_seconds"] += time.perf_counter() - started
        compute = _current_stream(self.device)
        # Later compute reads the new rows only after the transfers, and the
        # retired rows may be rewritten only after this forward's reads.
        _stream_wait_event(compute, event)
        retire_fence = _record_event(compute)
        touched = []
        for layer_index, swap, vram_spare, ram_spare in transaction.entries:
            layer = self.layers[layer_index]
            layer.flip_swap(swap, vram_spare, ram_spare, retire_fence)
            self.stats["swaps"] += 1
            self.per_layer_swaps[layer_index] += 1
            self.stats["h2d_bytes"] += layer.row_bytes
            self.stats["d2h_bytes"] += layer.row_bytes
            if layer not in touched:
                touched.append(layer)
        for layer in touched:
            layer.publish_maps()
        started = time.perf_counter()
        self.policy.commit(transaction.plan)
        self.stats["resyncs"] += 1
        self.stats["async_commits"] += 1
        for i, layer in enumerate(self.layers):
            if (
                tuple(self.policy.expert_to_hot[i]) != layer.hot_map_host
                or tuple(self.policy.expert_to_cold[i]) != layer.cold_map_host
            ):
                raise AssertionError("Policy and physical tier maps diverged")
        self.stats["policy_commit_seconds"] += time.perf_counter() - started
        self.stats["async_flip_delay_seconds"] += (
            time.perf_counter() - transaction.enqueued_at
        )
        rebase = getattr(self.observer, "rebase", None)
        if rebase is not None:
            rebase(
                tokens_total=self.policy.tokens_total,
                version=self.policy.version,
                last_sync_tokens=self.policy.last_sync_tokens,
            )
        transaction.state = "committed"
        self.pending = None
        return True

    def migrate(self, swaps):
        """Move a plan's rows in slot-independent waves, in plan order.

        Each wave takes two stream waits instead of two per swap; the host
        maps advance per swap in plan order, and each touched layer's device
        maps are republished once after its last wave completed.
        """
        synchronize = _current_stream(self.device).synchronize
        touched = []
        for wave in plan_waves(swaps, self.settings.temp_slots):
            items = []
            for index, swap in enumerate(wave):
                layer = self.layers[swap.layer]
                layer.stage_swap(
                    swap.old_expert, swap.new_expert, swap.hot_slot, swap.cold_slot
                )
                resolve_hot = getattr(layer, "resolve_hot_row", lambda slot: slot)
                resolve_cold = getattr(layer, "resolve_cold_row", lambda slot: slot)
                items.append(
                    (
                        getattr(layer, "bank", layer.hot),
                        layer.cold_cpu,
                        temporary_row(self.temporary, index),
                        resolve_hot(swap.hot_slot),
                        resolve_cold(swap.cold_slot),
                    )
                )
                if layer not in touched:
                    touched.append(layer)
            swap_tensor_rows_wave(items, synchronize)
            for swap in wave:
                layer = self.layers[swap.layer]
                self.stats["swaps"] += 1
                self.per_layer_swaps[swap.layer] += 1
                self.stats["h2d_bytes"] += layer.row_bytes
                self.stats["d2h_bytes"] += layer.row_bytes
                self.stats["host_copy_bytes"] += layer.row_bytes
            self.stats["migration_waves"] += 1
            self.stats["max_wave_swaps"] = max(self.stats["max_wave_swaps"], len(wave))
        for layer in touched:
            layer.publish_maps()

    def report(self):
        if self.settings.moe_kernel == "native":
            # The adapter records missing or out-of-range routes on the
            # device; one host read per report, as for the promote tables.
            for workspace in (
                *_NATIVE_WORKSPACES.values(),
                *_NATIVE_PREFILL_WORKSPACES.values(),
            ):
                if int(workspace.error.reshape(-1)[0].item()):
                    self.poisoned = True
                    raise RuntimeError("Native adapter recorded a routing error")
        if self.settings.promote:
            # The device placement is the truth: validate it and refresh the
            # host maps for the report. One host copy per report, not per step.
            pool = getattr(self, "pool", None)
            if pool is not None:
                if self.settings.control_file:
                    applied = pool.poll_control_file(self.settings.control_file)
                    if applied:
                        LOGGER.warning(
                            "LAB_EXPERT_TIER_CONTROL %s",
                            json.dumps(applied, sort_keys=True),
                        )
                    rejected = pool.control_error
                    if rejected and rejected != self._control_rejected:
                        # One line per rejected file version; values unchanged.
                        LOGGER.warning("LAB_EXPERT_TIER_CONTROL_REJECTED %s", rejected)
                    self._control_rejected = rejected
                self.stats["pool_resident_per_layer"] = pool.snapshot()
                self.stats["pool_control"] = pool.control()
            for layer in self.layers:
                if (
                    pool is not None
                    or getattr(layer, "promote_tables", None) is not None
                ):
                    layer.promote_snapshot()
        # Defaults make snapshots/deltas stable even before the first swap.
        fields = (
            "model_forwards",
            "model_tokens",
            "route_hot",
            "route_total",
            "swaps",
            "h2d_bytes",
            "d2h_bytes",
            "host_copy_bytes",
            "resyncs",
            "migration_waves",
            "max_wave_swaps",
            "async_plans",
            "async_commits",
            "async_swaps_enqueued",
            "async_pending_boundaries",
            "async_wait_seconds",
            "async_enqueue_seconds",
            "async_flip_delay_seconds",
            "sync_fallbacks",
            "promote_steps_observed",
            "recorded_forwards",
            "replayed_forwards",
            "captured_forwards",
            "dropped_startup_records",
            "deferred_forwards",
            "snapshot_forwards",
            "device_snapshots",
            "policy_observe_seconds",
            "policy_plan_seconds",
            "policy_commit_seconds",
            "migration_wall_seconds",
        )
        snapshot: dict[str, Any] = {key: self.stats[key] for key in fields}
        snapshot.update(self.stats)
        # Hit rate is measured by the host readback observer, or by a device
        # snapshot that says it counted from real maps; never present an
        # unmeasured zero as 0%.
        route_hot_available = (
            bool(getattr(self.observer, "reports_route_hot", False))
            or self.route_hot_measured
        )
        if not route_hot_available:
            snapshot["route_hot"] = None
        snapshot["route_hot_available"] = route_hot_available
        snapshot["draft_resident"] = self.draft_resident
        snapshot.update(
            {
                "timestamp_ns": time.time_ns(),
                "tokens_total": self.policy.tokens_total,
                "version": self.policy.version,
                "heat_enabled": self.heat_enabled,
                "pending_layers": self.recorded,
                "unfinished_forward_rows": self.forward_rows,
                "async_pending": self.pending is not None,
                "promote_mode": self.settings.promote,
                "global_pool": self.settings.global_pool,
                "per_layer_swaps": list(self.per_layer_swaps),
                "policy_cpu_seconds": sum(
                    self.stats[key]
                    for key in (
                        "policy_observe_seconds",
                        "policy_plan_seconds",
                        "policy_commit_seconds",
                    )
                ),
                "settings": asdict(self.settings),
                "policy_config": self.settings.policy_kwargs(),
                "route_counts_before_migration": True,
            }
        )
        LOGGER.warning("LAB_EXPERT_TIER_STATS %s", json.dumps(snapshot, sort_keys=True))

    def enable_heat(self):
        with self.lock:
            if self.poisoned or self.recorded:
                raise RuntimeError(
                    "Cannot enable tier heat with failed or incomplete startup work"
                )
            if self.heat_enabled:
                raise RuntimeError("Tier heat cannot be enabled twice")
            if self.policy.tokens_total or self.policy.swaps_total:
                raise AssertionError("Startup polluted tier heat/migration accounting")
            if self.forward_rows is not None:
                # The last direct warmup/capture forward is startup work.
                self.discard_unconsumed("heat enabled")
            self.settle_pending(wait=True)
            if self.stream is not None:
                self.stream.synchronize()
            startup = dict(self.stats)
            self.stats.clear()
            self.per_layer_swaps[:] = [0] * len(self.layers)
            self.heat_enabled = True
            # Captured graphs cannot see this Python flag: the observer must
            # flip its own in-place device gate now, after the startup wait.
            self.observer.on_heat_enabled()
            if self.settings.promote:
                for layer in self.layers:
                    layer.set_promote_gate(True)
            LOGGER.warning(
                "LAB_EXPERT_TIER_HEAT_ENABLED %s",
                json.dumps(
                    {
                        "heat_enabled": True,
                        "tokens_total": self.policy.tokens_total,
                        "swaps_total": self.policy.swaps_total,
                        "pending_layers": self.recorded,
                        "startup_stats": startup,
                    },
                    sort_keys=True,
                ),
            )


def enable_model_heat(model):
    """Called only after the worker's entire startup/warmup succeeds."""
    if Settings.from_env() is None:
        return
    coordinator = getattr(model, "_lab_expert_tier_coordinator", None)
    if coordinator is None:
        raise RuntimeError("Enabled tier model has no coordinator after startup")
    coordinator.enable_heat()


def finish_model_forward(model, rows, valid_rows=None, is_decode=None):
    """Runner hook after every model forward: eager, dummy, or graph replay.

    Cheap when the tier is disabled; never reads the environment. `is_decode`
    is the runner's "no prefill in this forward" (None when not stated).
    """
    coordinator = getattr(model, "_lab_expert_tier_coordinator", None)
    if coordinator is not None:
        coordinator.finish_forward(rows, valid_rows, is_decode)


def unpack_routes(packed, num_experts):
    """Validate the one-copy GPU routing record before updating global heat."""
    if not packed or not packed[0] or not packed[0][0]:
        raise ValueError("Empty model routing record")
    rows, width = len(packed[0]), len(packed[0][0])
    if width < 3 or width % 2 != 1:
        raise ValueError("Malformed packed routing width")
    k = (width - 1) // 2
    mask = [bool(row[-1]) for row in packed[0]]
    routes, activity = [], []
    for layer in packed:
        if len(layer) != rows or any(len(row) != width for row in layer):
            raise ValueError("Layer routing shapes differ within model forward")
        if any(row[-1] not in (0, 1) for row in layer):
            raise ValueError("Invalid model routing valid-mask value")
        if [bool(row[-1]) for row in layer] != mask:
            raise ValueError("Layer padding masks differ within model forward")
        ids, weights = [], []
        for row_index, row in enumerate(layer):
            row_ids = row[:k]
            for expert in row_ids:
                if 0 <= expert < num_experts:
                    continue
                if expert == -1 and not mask[row_index]:
                    continue
                raise ValueError(
                    f"Invalid expert ID {expert} on routing row {row_index}; "
                    "only validated padding rows may contain -1"
                )
            ids.append(row_ids)
            weights.append(row[k : 2 * k])
        if any(value not in (0, 1) for row in weights for value in row):
            raise ValueError("Invalid routing activity flag")
        routes.append(ids)
        activity.append(weights)
    return routes, activity, mask, sum(mask)


def _validate_sources(name, layer):
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    sources = {}
    for tensor_name in TENSORS:
        parameter = getattr(layer, tensor_name)
        cpu = _get_cpu_source(parameter)
        if not getattr(parameter, "_vllm_is_uva_offloaded", False) or cpu is None:
            raise RuntimeError(
                f"{name}.{tensor_name}: requires all expert tensors in UVA; "
                "use --cpu-offload-gb 80 --cpu-offload-params experts"
            )
        if (
            cpu.device.type != "cpu"
            or not cpu.is_pinned()
            or not cpu.is_contiguous()
            or parameter.device.type != "cuda"
            or cpu.shape != parameter.shape
            or cpu.dtype != parameter.dtype
            or cpu.ndim < 1
            or cpu.shape[0] != layer.global_num_experts
            or not parameter.is_contiguous()
        ):
            raise RuntimeError(
                f"{name}.{tensor_name}: invalid final pinned source/UVA layout"
            )
        if get_accelerator_view_from_cpu_tensor(cpu).data_ptr() != parameter.data_ptr():
            raise RuntimeError(
                f"{name}.{tensor_name}: CPU owner does not back UVA Parameter"
            )
        sources[tensor_name] = cpu
    return sources


SPECULATION_METHODS = ("ngram", "ngram_gpu", "mtp")


def check_speculation(speculative_config, spec_rows):
    """Admit vLLM speculation only in the agreed first form.

    n-gram has no draft model; MTP loads its draft under `draft_load_scope`
    so the draft's layers stay outside the tier. Other draft-model methods
    (DFlash, EAGLE, ...) are rejected until their loaders do the same. Every
    verify step has 1 + num_speculative_tokens rows per request and must fit
    the pool decode path (`spec_rows`).
    """
    if speculative_config is None:
        return
    method = getattr(speculative_config, "method", None)
    if method not in SPECULATION_METHODS:
        raise NotImplementedError(
            f"Expert tier admits speculation with methods {SPECULATION_METHODS},"
            f" got {method!r}"
        )
    tokens = int(getattr(speculative_config, "num_speculative_tokens", 0) or 0)
    if tokens < 1 or tokens + 1 > spec_rows:
        raise NotImplementedError(
            f"num_speculative_tokens={tokens} needs SPEC_ROWS >= {tokens + 1}"
        )


def _compact_one(
    index, name, layer, method, slots, settings, temporary, pool=None, max_tokens=None
):
    sources = _validate_sources(name, layer)
    refs = tuple(weakref.ref(tensor) for tensor in sources.values())
    original = {key: getattr(layer, key) for key in TENSORS}
    original_kernel = method.moe_kernel
    if settings.moe_kernel == "native":
        if original_kernel is not None or not getattr(method, "_lab_native", False):
            raise RuntimeError(f"{name}: loader did not keep the native layout")
    else:
        _check_kernel_scales(original_kernel, original)
    tier = TierLayer(
        index,
        name,
        layer,
        method,
        sources,
        slots,
        settings,
        pool,
        max_tokens=max_tokens,
    )
    if settings.verify_init:
        tier.verify_initial(original_kernel, original, temporary)
    replace_full_source_references(
        layer, method, tier.hot, tier.hot_kernel, tier.hot_quant
    )
    # No original kernel, Parameters, or full source dictionaries escape.
    return tier, refs


DRAFT_PREFIXES = {"mtp": "mtp."}


def reserve_draft(speculative_config, settings, tier_bytes):
    """Estimate the draft's checkpoint bytes before it loads (None without a
    draft model) and, when VRAM_BUDGET_GIB is set, check tier + draft fit."""
    method = getattr(speculative_config, "method", None)
    prefix = DRAFT_PREFIXES.get(str(method)) if method is not None else None
    if prefix is None:
        return None
    from .draft_capacity import estimate_draft_bytes

    draft_config = speculative_config.draft_model_config
    estimate = estimate_draft_bytes(draft_config.model, prefix)
    estimate["checkpoint"] = draft_config.model
    budget = int(settings.vram_budget_gib * 2**30)
    estimate["budget_bytes"] = budget or None
    estimate["tier_bytes"] = int(tier_bytes)
    LOGGER.warning("LAB_EXPERT_TIER_DRAFT_RESERVE %s", json.dumps(estimate))
    if budget and tier_bytes + estimate["bytes"] > budget:
        raise RuntimeError(
            f"Tier bytes {tier_bytes} + draft reservation {estimate['bytes']} "
            f"exceed VRAM_BUDGET_GIB ({budget} bytes)"
        )
    return estimate


def record_draft_model(target_model, draft_model):
    """Draft loader hook after the draft is loaded and shares the target's
    embedding/head: measure unique resident bytes against the reservation.

    No-op without the tier. Shared storage is reported, never charged.
    """
    coordinator = getattr(target_model, "_lab_expert_tier_coordinator", None)
    if coordinator is None:
        return None
    from .draft_capacity import check_estimate, measure_resident_bytes

    measured = measure_resident_bytes(draft_model, shared_with=target_model)
    estimate = coordinator.draft_reserve
    # The measurement is logged before any check so a rejected load still
    # leaves its breakdown (by dtype, against the header estimate by dtype).
    LOGGER.warning(
        "LAB_EXPERT_TIER_DRAFT_MEASURED %s",
        json.dumps({**measured, "estimate": estimate}),
    )
    if estimate is None:
        raise RuntimeError("Draft model loaded without a tier reservation")
    result = {
        **measured,
        **check_estimate(estimate, measured, coordinator.settings.draft_tolerance),
    }
    coordinator.draft_resident = result
    LOGGER.warning("LAB_EXPERT_TIER_DRAFT_RESIDENT %s", json.dumps(result))
    return result


def initialize_model(model, model_config):
    from .draft_scope import is_draft_load_scope

    if is_draft_load_scope():
        # Draft models (MTP, EAGLE) are never tier targets.
        return
    settings = Settings.from_env()
    if settings is None:
        return
    from .promote import configure_copy

    configure_copy(
        settings.copy_shape,
        settings.copy_programs or None,
        settings.copy_words,
    )
    import torch

    from vllm import envs
    from vllm.config import CompilationMode, get_current_vllm_config
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        is_workspace_manager_initialized,
    )

    config = get_current_vllm_config()
    if not envs.VLLM_MOE_SKIP_PADDING:
        raise NotImplementedError(
            "Tier requires VLLM_MOE_SKIP_PADDING=1 for valid MRv2 heat accounting"
        )
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
        raise NotImplementedError("Tier requires VLLM_USE_V2_MODEL_RUNNER=1")
    compilation = config.compilation_config
    graph_mode = compilation.cudagraph_mode
    if compilation.mode != CompilationMode.NONE:
        raise NotImplementedError(
            "Expert tier requires compilation mode NONE (no torch.compile); use "
            "--enforce-eager, or -cc.mode=none with cudagraph_mode=FULL_DECODE_ONLY"
        )
    if envs.VLLM_USE_BREAKABLE_CUDAGRAPH or (
        graph_mode is not None and graph_mode.has_piecewise_cudagraphs()
    ):
        raise NotImplementedError(
            "Expert tier supports only NONE, FULL_DECODE_ONLY, or FULL CUDA graphs"
        )
    # Graph padding never exceeds the capture size; eager prefill never
    # exceeds the scheduler budget. Size the static routing records for both.
    max_tokens = config.scheduler_config.max_num_batched_tokens
    if compilation.max_cudagraph_capture_size:
        max_tokens = max(max_tokens, compilation.max_cudagraph_capture_size)
    if settings.verify_init and (
        not is_workspace_manager_initialized()
        or current_workspace_manager().is_locked()
    ):
        raise RuntimeError(
            "Init verification requires the existing unlocked workspace manager"
        )
    if config.lora_config is not None:
        raise NotImplementedError("LoRA is unsupported")
    check_speculation(config.speculative_config, settings.spec_rows)
    if config.parallel_config.pipeline_parallel_size != 1:
        raise NotImplementedError("Pipeline parallelism is unsupported")
    candidates, row_sizes = [], []
    for name, layer in model.named_modules():
        method = getattr(layer, "quant_method", None)
        if not isinstance(method, ModelOptNvFp4FusedMoE):
            continue
        if hasattr(method, "_lab_expert_tier"):
            raise RuntimeError("Expert tier cannot be initialized/reloaded twice")
        parallel = method.moe.moe_parallel_config
        if any(
            getattr(parallel, key) != 1
            for key in ("tp_size", "dp_size", "ep_size", "pcp_size", "sp_size")
        ):
            raise NotImplementedError("Expert tier requires TP=DP=EP=PCP=SP=1")
        if settings.moe_kernel == "native":
            if not getattr(method, "_lab_native", False):
                raise RuntimeError(f"{name}: native backend layout was not loaded")
            from .native_loader import require_silu

            require_silu(getattr(layer, "activation", "silu"))
        elif method.nvfp4_backend != NvFp4MoeBackend.MARLIN or method.is_monolithic:
            raise NotImplementedError("Only modular NVFP4 Marlin is supported")
        if layer.expert_map is not None or layer.global_num_experts != 512:
            raise NotImplementedError("Expected unsharded 512-expert source banks")
        if (
            method.moe.is_lora_enabled
            or method.moe.has_bias
            or layer.apply_router_weight_on_input
        ):
            raise NotImplementedError("LoRA, bias, and router-on-input are unsupported")
        if method.moe.in_dtype != torch.bfloat16:
            raise NotImplementedError("Expert tier requires BF16 activations")
        if layer.w13_input_scale is not None or layer.w2_input_scale is not None:
            raise RuntimeError("Marlin must remove input quantization scales")
        # Validate all sources now, but retain no CPU source dict in candidates.
        src = _validate_sources(name, layer)
        row_sizes.append(sum(t[0].numel() * t.element_size() for t in src.values()))
        del src
        candidates.append((name, layer, method))
    if len(candidates) != 48:
        raise NotImplementedError(
            f"Expected all 48 FlashNext MoE layers, found {len(candidates)}"
        )
    staging_rows = (
        candidates[0][2].moe.experts_per_token * settings.spec_rows
        if settings.staging
        else 0
    )
    spare_rows = (
        settings.temp_slots if (settings.async_migration or settings.promote) else 0
    )
    if settings.promote and settings.planner == "device":
        import importlib.util

        if importlib.util.find_spec("vllm._lab_expert_tier.device_lru") is None:
            raise NotImplementedError(
                "PROMOTE=1 with PLANNER=device needs vllm._lab_expert_tier.device_lru"
            )
    explicit = (
        None
        if settings.layer_slots == "uniform"
        else [int(item) for item in settings.layer_slots.split(",")]
    )
    pool = None
    if settings.global_pool:
        # One bank for every layer: identical rows are required. Shared
        # staging rows are charged once; every other byte is pool rows,
        # distributed uniformly (or as LAYER_SLOTS) as the starting
        # placement that the LRU then rebalances.
        if len(set(row_sizes)) != 1:
            raise NotImplementedError("Global pool requires identical layer rows")
        staging_bytes = staging_rows * row_sizes[0]
        slots_per_layer, expected_bytes = allocate_slots(
            settings.capacity_bytes - staging_bytes,
            row_sizes,
            512,
            reserve=0,
            layer_slots=explicit,
        )
        expected_bytes += staging_bytes
    else:
        slots_per_layer, expected_bytes = allocate_slots(
            settings.capacity_bytes,
            row_sizes,
            512,
            reserve=staging_rows + spare_rows,
            layer_slots=explicit,
        )
    if any(not 0 < slots < 512 for slots in slots_per_layer):
        raise ValueError("Expert tier requires both a hot and cold partition")
    if settings.verify_init:
        check_verify_capacity(
            slots_per_layer, 512, candidates[0][2].moe.experts_per_token
        )
    first = candidates[0][1]
    temporary = {
        name: torch.empty(
            (settings.temp_slots, *getattr(first, name).shape[1:]),
            dtype=getattr(first, name).dtype,
            device="cpu",
            pin_memory=True,
        )
        for name in TENSORS
    }
    if settings.global_pool:
        from .global_pool import GlobalPool

        pool = GlobalPool(
            first.w13_weight.device,
            _validate_sources(candidates[0][0], first),
            slots_per_layer,
            staging_rows,
        )
        pool.apply_control(
            promote_limit=settings.promote_limit,
            promote_interval=settings.promote_interval,
            promote_min_misses=settings.promote_min_misses,
            protect_recent=settings.protect_recent,
        )
    tiers = []
    for index, (name, layer, method) in enumerate(candidates):
        tier, raw_refs = _compact_one(
            index,
            name,
            layer,
            method,
            slots_per_layer[index],
            settings,
            temporary_row(temporary, 0),
            pool,
            max_tokens=max_tokens,
        )
        gc.collect()
        if not settings.ram_backing and any(ref() is not None for ref in raw_refs):
            raise RuntimeError(
                f"{name}: original host source remains referenced after compaction"
            )
        tiers.append(tier)
        LOGGER.warning(
            "LAB_EXPERT_TIER_COMPACT %s",
            json.dumps(
                {
                    "layer": name,
                    "index": index,
                    "raw_host_owners_released": not settings.ram_backing,
                    "source_bank_retained": settings.ram_backing,
                    "hot_bytes": tier.hot_bytes,
                    "cold_bytes": tier.cold_bytes,
                    "host_bytes": tier.host_bytes,
                    "host_allocator": _host_allocator_stats(),
                },
                sort_keys=True,
            ),
        )
    coordinator = TierCoordinator(
        tiers,
        settings,
        temporary,
        make_observer(
            settings.observer,
            num_layers=len(tiers),
            num_experts=tiers[0].num_experts,
            decay=settings.decay,
            sync_period=settings.sync_tokens,
            max_step_tokens=settings.spec_rows,
            session_id=os.getpid(),
        ),
    )
    coordinator.allocate_records(
        tiers[0].device, candidates[0][2].moe.experts_per_token, max_tokens
    )
    for tier in tiers:
        tier.coordinator = coordinator
        tier.method._lab_expert_tier = tier
    model._lab_expert_tiers = tiers
    model._lab_expert_tier_coordinator = coordinator
    coordinator.pool = pool
    coordinator.draft_reserve = reserve_draft(
        config.speculative_config, settings, expected_bytes
    )
    atexit.register(coordinator.report)
    atexit.register(coordinator.flush)  # LIFO: deliver deferred work first
    actual_bytes = sum(t.hot_bytes + t.staging_bytes + t.spare_bytes for t in tiers)
    if pool is not None:
        actual_bytes += pool.staging_bytes
        if pool.pool_bytes != sum(t.hot_bytes for t in tiers):
            raise AssertionError("Pool rows and layer hot rows disagree")
    if actual_bytes != expected_bytes or actual_bytes > settings.capacity_bytes:
        raise AssertionError("Tier exceeds exact six-tensor GPU budget")
    LOGGER.warning(
        "LAB_EXPERT_TIER_READY %s",
        json.dumps(
            {
                "layers": len(tiers),
                "experts_per_layer": 512,
                "hot_slots_per_layer": (
                    slots_per_layer[0]
                    if len(set(slots_per_layer)) == 1
                    else list(slots_per_layer)
                ),
                "cold_slots_per_layer": (
                    512 - slots_per_layer[0]
                    if len(set(slots_per_layer)) == 1
                    else [512 - slots for slots in slots_per_layer]
                ),
                "layer_slots": settings.layer_slots,
                "capacity_bytes": settings.capacity_bytes,
                "gpu_weight_bytes": actual_bytes,
                "staging_slots_per_layer": staging_rows,
                "staging_bytes": (
                    pool.staging_bytes
                    if pool is not None
                    else sum(t.staging_bytes for t in tiers)
                ),
                "spare_rows_per_layer": 0 if pool is not None else spare_rows,
                "spare_bytes": sum(t.spare_bytes for t in tiers),
                "host_cold_bytes": sum(t.cold_bytes for t in tiers),
                "host_cold_spare_bytes": sum(t.cold_spare_bytes for t in tiers),
                "host_bytes": sum(t.host_bytes for t in tiers),
                "ram_backing": settings.ram_backing,
                "global_pool": settings.global_pool,
                "pool_rows": None if pool is None else pool.tables.pool_rows,
                "async_migration": settings.async_migration,
                "promote_mode": settings.promote,
                "planner": settings.planner if settings.promote else None,
                "temporary_host_bytes": sum(
                    t.numel() * t.element_size() for t in temporary.values()
                ),
                "temporary_rows": settings.temp_slots,
                "split": settings.split,
                "observer": settings.observer,
                "host_source_bytes": sum(row_sizes) * 512,
                "host_allocator": _host_allocator_stats(),
                "verify_init": settings.verify_init,
                "policy": "expert_tier_heat_periodic_ram_temp",
                "cuda_graphs": None if graph_mode is None else graph_mode.name,
                "compilation_mode": CompilationMode(compilation.mode).name,
                "routing_record_tokens": max_tokens,
                "routing_record_bytes": coordinator.records.numel() * 4,
                "static_maps_and_records": True,
                "settings": asdict(settings),
                "policy_config": settings.policy_kwargs(),
                "static_partition": settings.sync_tokens == 0,
                "source_bank_retained": settings.ram_backing,
                "moe_kernel": settings.moe_kernel,
                "native_prefill": settings.native_prefill,
                "native_gemv_rows": settings.native_gemv_rows,
                "prefill_stage_rows": settings.prefill_stage_rows,
                "prefill_stage_bytes": (
                    settings.prefill_stage_rows * row_sizes[0]
                    if settings.prefill_stage_rows
                    else 0
                ),
                "record_kernel": settings.record_kernel,
                "shared_gate": settings.shared_gate,
                "native_output": settings.native_output,
                "copy_shape": settings.copy_shape,
                "spec_rows": settings.spec_rows,
                "draft_reserve": coordinator.draft_reserve,
                "pool_control": None if pool is None else pool.control(),
                "control_file": settings.control_file or None,
                "verify_file": settings.verify_file or None,
                "copy_programs": settings.copy_programs or None,
                "copy_words": settings.copy_words,
                "routing_host_copies_per_model_step": 1,
            },
            sort_keys=True,
        ),
    )
