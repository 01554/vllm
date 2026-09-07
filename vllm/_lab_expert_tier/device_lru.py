# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, device-side LRU admission for promote mode.

The planner is deliberately separate from the promote flip.  It only reads
the placement maps and emits fixed-size copy plans; the flip owns all map
updates.  The CPU implementation below is the reference used by tests and by
non-CUDA callers.  CUDA uses one Triton program so the plan remains safe to
capture in a graph.

The victim scan is adapted from the sequential arg-min approach in
``flashlib.kernels.slot_cache.triton.lru_ensure`` (Apache-2.0).  That kernel's
map-install step is intentionally not used here: promote's flip owns map
installation and the planner must leave both owner maps untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_KERNELS: dict[str, Any] = {}


@dataclass
class LRUState:
    """Persistent planner storage allocated before CUDA graph capture."""

    width: int
    gate: Any
    pending: Any
    last_use: Any
    clock: Any
    error: Any
    promote_expert: Any
    promote_cold_slot: Any
    victim_expert: Any
    victim_hot_slot: Any
    count: Any
    staged_only_expert: Any
    staged_only_cold_slot: Any
    staged_only_count: Any

    @property
    def gate_open(self):
        """Alias retained for callers that describe the gate semantically."""
        return self.gate


@dataclass(frozen=True)
class StepPlan:
    """Fixed-shape output of :func:`plan_step`.

    Every vector has exactly the configured staging width.  ``count`` and
    ``staged_only_count`` describe the valid prefixes of their respective
    vectors.
    """

    promote_expert: Any
    promote_cold_slot: Any
    victim_expert: Any
    victim_hot_slot: Any
    count: Any
    staged_only_expert: Any
    staged_only_cold_slot: Any
    staged_only_count: Any


def _next_power_of_two(value: int) -> int:
    value = max(int(value), 1)
    return 1 << (value - 1).bit_length()


def _state_plan(state: LRUState) -> StepPlan:
    return StepPlan(
        promote_expert=state.promote_expert,
        promote_cold_slot=state.promote_cold_slot,
        victim_expert=state.victim_expert,
        victim_hot_slot=state.victim_hot_slot,
        count=state.count,
        staged_only_expert=state.staged_only_expert,
        staged_only_cold_slot=state.staged_only_cold_slot,
        staged_only_count=state.staged_only_count,
    )


def allocate_state(tables: Any, width: int) -> LRUState:
    """Allocate fixed-address planner state and attach it to ``tables``.

    Args:
        tables: Promote tables containing the owner maps and LRU aliases.
        width: Number of output lanes reserved for one layer.

    Returns:
        The newly allocated :class:`LRUState`.  The same object is also stored
        as ``tables.lru_state``.

    Raises:
        ValueError: If ``width`` is negative or the required table fields are
            missing.
    """
    import torch

    if not isinstance(width, int) or isinstance(width, bool) or width < 0:
        raise ValueError("planner width must be a non-negative Python integer")
    _check_tables(tables)
    device = tables.hot_map.device
    state = LRUState(
        width=width,
        gate=torch.zeros(1, dtype=torch.int32, device=device),
        pending=torch.zeros(1, dtype=torch.int32, device=device),
        # These are aliases, not copies.  The planner and the caller therefore
        # see one recency/error state even when tables are inspected later.
        last_use=tables.last_use,
        clock=tables.clock,
        error=tables.error,
        promote_expert=torch.zeros(width, dtype=torch.int64, device=device),
        promote_cold_slot=torch.zeros(width, dtype=torch.int64, device=device),
        victim_expert=torch.zeros(width, dtype=torch.int64, device=device),
        victim_hot_slot=torch.zeros(width, dtype=torch.int64, device=device),
        count=torch.zeros(1, dtype=torch.int32, device=device),
        staged_only_expert=torch.zeros(width, dtype=torch.int64, device=device),
        staged_only_cold_slot=torch.zeros(width, dtype=torch.int64, device=device),
        staged_only_count=torch.zeros(1, dtype=torch.int32, device=device),
    )
    tables.lru_state = state
    return state


def open_gate(state: LRUState) -> None:
    """Enable promotion and recency updates in-place on the device."""
    state.gate.fill_(1)


def close_gate(state: LRUState) -> None:
    """Disable promotion and recency updates in-place on the device."""
    state.gate.zero_()


def set_pending(state: LRUState, pending: bool) -> None:
    """Temporarily suppress promotion while retaining open-gate recency.

    Runtime code normally leaves this optional gate at zero.  It is useful for
    callers that have a transfer pending: selected experts still age the LRU,
    while all misses remain staging-only until the pending work is settled.
    """
    state.pending.fill_(1 if pending else 0)


def _check_tables(tables: Any) -> None:
    import torch

    required = (
        "hot_map",
        "cold_map",
        "last_use",
        "clock",
        "error",
        "vram_free",
        "ram_free",
    )
    missing = [name for name in required if not hasattr(tables, name)]
    if missing:
        raise ValueError("planner tables are missing " + ", ".join(missing))
    hot_map = tables.hot_map
    cold_map = tables.cold_map
    if hot_map.ndim != 1 or cold_map.ndim != 1:
        raise ValueError("hot_map and cold_map must be one-dimensional")
    if hot_map.shape != cold_map.shape:
        raise ValueError("hot_map and cold_map must have the same shape")
    if tables.last_use.ndim != 1 or tables.last_use.shape != hot_map.shape:
        raise ValueError("last_use must have one entry per expert")
    for name in ("clock", "error"):
        value = getattr(tables, name)
        if value.ndim != 1 or value.shape != (1,):
            raise ValueError(f"{name} must have shape [1]")
    for name in ("vram_free", "ram_free"):
        if getattr(tables, name).ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    if hot_map.dtype != torch.int32:
        raise ValueError("owner maps must be int32")
    if cold_map.dtype != torch.int32:
        raise ValueError("owner maps must be int32")
    if tables.last_use.dtype != torch.int64:
        raise ValueError("last_use must be int64")
    if tables.clock.dtype != torch.int64:
        raise ValueError("clock must be int64")
    if tables.error.dtype != torch.int32:
        raise ValueError("error must be int32")
    tensors = [
        hot_map,
        cold_map,
        tables.last_use,
        tables.clock,
        tables.error,
        tables.vram_free,
        tables.ram_free,
    ]
    device = hot_map.device
    if any(t.device != device for t in tensors):
        raise ValueError("planner tables must share one device")


def _validate_ids(ids: Any, tables: Any, width: int) -> None:
    import torch

    if not isinstance(width, int) or isinstance(width, bool) or width < 0:
        raise ValueError("planner width must be a non-negative Python integer")
    if not isinstance(ids, torch.Tensor):
        raise TypeError("ids must be a torch.Tensor")
    if ids.ndim != 2:
        raise ValueError("ids must have shape [rows, k]")
    if ids.dtype not in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("ids must have an integral dtype")
    if ids.numel() > width:
        raise ValueError("ids contain more lanes than the planner width")
    if ids.device != tables.hot_map.device:
        raise ValueError("ids and planner tables must share one device")
    _check_tables(tables)


def plan_step(ids: Any, tables: Any, width: int) -> StepPlan:
    """Plan promotions and staging for one layer call.

    ``ids`` is read-only.  ``-1`` is padding; any other out-of-range ID records
    a sticky device error and is excluded.  A valid expert whose two owner maps
    are both ``-1`` is treated the same way.  Distinct misses are sorted by
    expert ID.  Victims are all unselected hot experts ordered by
    ``(last_use, logical_hot_slot)``.
    """
    _validate_ids(ids, tables, width)
    state = getattr(tables, "lru_state", None)
    if state is None:
        raise RuntimeError("allocate_state must run before plan_step")
    if state.width != width:
        raise ValueError("planner width differs from allocated state")
    if tables.hot_map.device.type != "cuda":
        return _plan_step_reference(ids, tables, state)
    if not ids.is_contiguous():
        raise ValueError("CUDA planner requires contiguous ids")
    return _plan_step_cuda(ids.view(-1), tables, state)


def _reset_outputs(state: LRUState) -> None:
    state.promote_expert.zero_()
    state.promote_cold_slot.zero_()
    state.victim_expert.zero_()
    state.victim_hot_slot.zero_()
    state.count.zero_()
    state.staged_only_expert.zero_()
    state.staged_only_cold_slot.zero_()
    state.staged_only_count.zero_()


def _plan_step_reference(ids: Any, tables: Any, state: LRUState) -> StepPlan:
    """Host reference; host scalar reads are confined to this CPU path."""
    _reset_outputs(state)
    experts = int(tables.hot_map.numel())
    hot_map = tables.hot_map.tolist()
    cold_map = tables.cold_map.tolist()
    values = ids.reshape(-1).tolist()
    selected: set[int] = set()
    misses: set[int] = set()
    bad = False

    for raw in values:
        expert = int(raw)
        if expert == -1:
            continue
        if expert < 0 or expert >= experts:
            bad = True
            continue
        hot_slot = int(hot_map[expert])
        cold_slot = int(cold_map[expert])
        if hot_slot < 0 and cold_slot < 0:
            bad = True
            continue
        # A hot owner wins if malformed input names both partitions.
        selected.add(expert)
        if hot_slot < 0 and cold_slot >= 0:
            misses.add(expert)

    if bad:
        tables.error.fill_(1)

    gate_open = bool(int(state.gate.reshape(-1).tolist()[0]))
    pending = bool(int(state.pending.reshape(-1).tolist()[0]))
    if gate_open:
        clock = int(tables.clock.reshape(-1).tolist()[0]) + 1
        tables.clock.fill_(clock)
        # Set membership makes duplicates idempotent by construction.
        for expert in selected:
            tables.last_use[expert] = clock
    else:
        clock = int(tables.clock.reshape(-1).tolist()[0])

    misses_sorted = sorted(misses)
    hot_values = tables.hot_map.tolist()
    last_use = tables.last_use.tolist()
    victims = sorted(
        (
            (int(last_use[expert]), int(hot_values[expert]), expert)
            for expert in range(experts)
            if int(hot_values[expert]) >= 0 and expert not in selected
        ),
        key=lambda value: (value[0], value[1]),
    )
    capacity = min(
        int(tables.vram_free.numel()),
        int(tables.ram_free.numel()),
        state.width,
    )
    count = 0
    if gate_open and not pending:
        count = min(len(misses_sorted), len(victims), capacity)

    promoted = misses_sorted[:count]
    staged = misses_sorted[count:]
    for lane, expert in enumerate(promoted):
        state.promote_expert[lane] = expert
        state.promote_cold_slot[lane] = cold_map[expert]
        state.victim_expert[lane] = victims[lane][2]
        state.victim_hot_slot[lane] = victims[lane][1]
    for lane, expert in enumerate(staged):
        state.staged_only_expert[lane] = expert
        state.staged_only_cold_slot[lane] = cold_map[expert]
    state.count[0] = count
    state.staged_only_count[0] = len(staged)
    # Keep this local alive for debuggers without accidentally changing the
    # semantics above; the clock is written through its alias already.
    del clock
    return _state_plan(state)


def _plan_step_cuda(ids: Any, tables: Any, state: LRUState) -> StepPlan:
    """Launch the one-program CUDA planner with fixed output addresses."""
    experts = int(tables.hot_map.numel())
    width = state.width
    # Keep one launch even for a zero-width query.  The launch still advances
    # an open gate's clock, as every layer call must do.
    block_width = _next_power_of_two(width)
    block_experts = 1024
    blocks = max(1, (experts + block_experts - 1) // block_experts)
    _plan_kernel()[(1,)](
        ids,
        ids.numel(),
        experts,
        tables.hot_map,
        tables.cold_map,
        state.last_use,
        state.clock,
        state.error,
        tables.vram_free.numel(),
        tables.ram_free.numel(),
        state.gate,
        state.pending,
        state.promote_expert,
        state.promote_cold_slot,
        state.victim_expert,
        state.victim_hot_slot,
        state.count,
        state.staged_only_expert,
        state.staged_only_cold_slot,
        state.staged_only_count,
        width,
        WIDTH=block_width,
        BLOCK_E=block_experts,
        NUM_BLOCKS=blocks,
        num_warps=4,
    )
    return _state_plan(state)


def _plan_kernel():
    """Return the single Triton program used by :func:`_plan_step_cuda`."""
    if "plan" in _KERNELS:
        return _KERNELS["plan"]
    from vllm.triton_utils import tl, triton

    @triton.jit
    def bounded_lru_plan(
        ids_ptr,
        n,
        num_experts,
        hot_map_ptr,
        cold_map_ptr,
        last_use_ptr,
        clock_ptr,
        error_ptr,
        vram_capacity,
        ram_capacity,
        gate_ptr,
        pending_ptr,
        promote_expert_ptr,
        promote_cold_ptr,
        victim_expert_ptr,
        victim_hot_ptr,
        count_ptr,
        staged_expert_ptr,
        staged_cold_ptr,
        staged_count_ptr,
        width,
        WIDTH: tl.constexpr,
        BLOCK_E: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
    ):
        lane = tl.arange(0, WIDTH)
        output_mask = lane < width
        present = lane < n
        raw = tl.load(ids_ptr + lane, mask=present, other=-1).to(tl.int64)
        valid = present & (raw >= 0) & (raw < num_experts)
        bad_active = present & (raw != -1) & (~valid)
        bad_any = tl.sum(bad_active.to(tl.int32), axis=0) > 0

        zero64 = tl.zeros((WIDTH,), dtype=tl.int64)
        tl.store(promote_expert_ptr + lane, zero64, mask=output_mask)
        tl.store(promote_cold_ptr + lane, zero64, mask=output_mask)
        tl.store(victim_expert_ptr + lane, zero64, mask=output_mask)
        tl.store(victim_hot_ptr + lane, zero64, mask=output_mask)
        tl.store(staged_expert_ptr + lane, zero64, mask=output_mask)
        tl.store(staged_cold_ptr + lane, zero64, mask=output_mask)

        gate = tl.load(gate_ptr) != 0
        pending = tl.load(pending_ptr) != 0
        old_clock = tl.load(clock_ptr)
        new_clock = old_clock + 1
        tl.store(clock_ptr, new_clock, mask=gate)

        miss_total = tl.zeros((), dtype=tl.int32)
        eligible_total = tl.zeros((), dtype=tl.int32)

        # The first pass emits every distinct miss in ascending expert order.
        # Because blocks are visited in ascending order and each block's lanes
        # are ascending, the output is globally sorted without a second sort.
        for block in range(NUM_BLOCKS):
            start = block * BLOCK_E
            expert_lane = start + tl.arange(0, BLOCK_E)
            expert_mask = expert_lane < num_experts
            hot = tl.load(hot_map_ptr + expert_lane, mask=expert_mask, other=-1).to(
                tl.int32
            )
            cold = tl.load(cold_map_ptr + expert_lane, mask=expert_mask, other=-1).to(
                tl.int32
            )
            selected = (
                tl.sum(
                    ((raw[:, None] == expert_lane[None, :]) & valid[:, None]).to(
                        tl.int32
                    ),
                    axis=0,
                )
                > 0
            )
            owner = (hot >= 0) | (cold >= 0)
            missing_owner = expert_mask & selected & (~owner)
            bad_any = bad_any | (tl.sum(missing_owner.to(tl.int32), axis=0) > 0)
            touch = gate & selected & owner & expert_mask
            tl.store(last_use_ptr + expert_lane, new_clock, mask=touch)
            miss = expert_mask & selected & (hot < 0) & (cold >= 0)
            local_rank = tl.cumsum(miss.to(tl.int32), axis=0) - 1
            tl.store(
                promote_expert_ptr + miss_total + local_rank,
                expert_lane.to(tl.int64),
                mask=miss,
            )
            tl.store(
                promote_cold_ptr + miss_total + local_rank,
                cold.to(tl.int64),
                mask=miss,
            )
            miss_total += tl.sum(miss.to(tl.int32), axis=0)

            eligible = expert_mask & (hot >= 0) & (~selected)
            eligible_total += tl.sum(eligible.to(tl.int32), axis=0)

        raw_limit = tl.minimum(miss_total, eligible_total)
        raw_limit = tl.minimum(raw_limit, vram_capacity)
        raw_limit = tl.minimum(raw_limit, ram_capacity)
        raw_limit = tl.minimum(raw_limit, width)
        limit = tl.where(gate & (~pending), raw_limit, 0)

        # Pick at most WIDTH victims.  Each iteration scans every expert, so
        # experts beyond the output width are still eligible victims.  Compare
        # both ordering fields separately so arbitrary logical slot
        # values cannot collide in a packed key.
        key_max = 0x7FFFFFFFFFFFFFFF
        for i in range(WIDTH):
            active = i < limit
            best_use = tl.full((), key_max, tl.int64)
            best_hot = tl.full((), key_max, tl.int64)
            best_expert = tl.full((), -1, tl.int64)
            previous_lane = tl.arange(0, WIDTH)
            previous = tl.load(
                victim_expert_ptr + previous_lane,
                mask=(previous_lane < i) & (previous_lane < width),
                other=-1,
            )
            for block in range(NUM_BLOCKS):
                start = block * BLOCK_E
                expert_lane = start + tl.arange(0, BLOCK_E)
                expert_mask = expert_lane < num_experts
                hot = tl.load(hot_map_ptr + expert_lane, mask=expert_mask, other=-1).to(
                    tl.int64
                )
                use = tl.load(
                    last_use_ptr + expert_lane, mask=expert_mask, other=key_max
                ).to(tl.int64)
                selected = (
                    tl.sum(
                        ((raw[:, None] == expert_lane[None, :]) & valid[:, None]).to(
                            tl.int32
                        ),
                        axis=0,
                    )
                    > 0
                )
                already = (
                    tl.sum(
                        (
                            (expert_lane[:, None].to(tl.int64) == previous[None, :])
                            & (previous_lane[None, :] < i)
                        ).to(tl.int32),
                        axis=1,
                    )
                    > 0
                )
                candidate = active & expert_mask & (hot >= 0) & (~selected) & (~already)
                candidate_use = tl.where(candidate, use, key_max)
                block_use = tl.min(candidate_use, axis=0)
                block_has_candidate = tl.sum(candidate.to(tl.int32), axis=0) > 0
                candidate_tie = candidate & (use == block_use)
                candidate_hot = tl.where(candidate_tie, hot, key_max)
                block_hot = tl.min(candidate_hot, axis=0)
                block_lane = tl.argmax(
                    (candidate_tie & (hot == block_hot)).to(tl.int32),
                    axis=0,
                )
                block_expert = (start + block_lane).to(tl.int64)
                block_hot = tl.where(block_has_candidate, block_hot, key_max)
                better = block_has_candidate & (
                    (block_use < best_use)
                    | ((block_use == best_use) & (block_hot < best_hot))
                )
                best_use = tl.where(better, block_use, best_use)
                best_hot = tl.where(better, block_hot, best_hot)
                best_expert = tl.where(better, block_expert, best_expert)
            valid_victim = active & (best_expert >= 0)
            tl.store(victim_expert_ptr + i, best_expert, mask=valid_victim)
            tl.store(victim_hot_ptr + i, best_hot, mask=valid_victim)

        tl.debug_barrier()
        staged_total = miss_total - limit
        output_lane = tl.arange(0, WIDTH)
        miss_mask = output_lane < miss_total
        miss_expert = tl.load(promote_expert_ptr + output_lane, mask=miss_mask, other=0)
        miss_cold = tl.load(promote_cold_ptr + output_lane, mask=miss_mask, other=0)
        staged_mask = output_lane < staged_total
        staged_source = output_lane + limit
        staged_expert = tl.load(
            promote_expert_ptr + staged_source,
            mask=staged_mask,
            other=0,
        )
        staged_cold = tl.load(
            promote_cold_ptr + staged_source,
            mask=staged_mask,
            other=0,
        )
        promote_mask = output_lane < limit
        tl.store(
            promote_expert_ptr + output_lane,
            tl.where(promote_mask, miss_expert, 0),
            mask=output_mask,
        )
        tl.store(
            promote_cold_ptr + output_lane,
            tl.where(promote_mask, miss_cold, 0),
            mask=output_mask,
        )
        tl.store(staged_expert_ptr + output_lane, staged_expert, mask=staged_mask)
        tl.store(staged_cold_ptr + output_lane, staged_cold, mask=staged_mask)
        tl.store(count_ptr, limit.to(tl.int32))
        tl.store(staged_count_ptr, staged_total.to(tl.int32))
        # Error is sticky.  There is one program, so an atomic is sufficient
        # and also preserves an error raised by an earlier layer call.
        tl.atomic_or(error_ptr, 1, mask=bad_any)

    _KERNELS["plan"] = bounded_lru_plan
    return bounded_lru_plan
