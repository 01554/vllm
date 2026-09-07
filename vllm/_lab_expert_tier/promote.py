# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Promote mode: per-token cache management on the device, FreeToken style.

In promote mode the placement is owned by a device LRU planner instead of
the periodic heat policy. Every batch-1 decode step, inside the captured
graph and on the compute stream, each layer:

1. plans (planner module): which selected cold experts are promoted this
   step, which unselected hot experts are evicted to make room, and which
   remaining misses are only staged for this step;
2. gathers the promoted and staged experts' rows from RAM into VRAM;
3. evicts the victims' rows from VRAM into RAM;
4. flips the device tables so the promoted experts are hot and the victims
   are cold, and builds the step's expert map;
5. runs the MoE kernels on the bank through that map.

All of it runs on one stream with fixed shapes, so ordering is by stream
order alone: reads of a row always precede the write that recycles it, and
no host synchronization is needed per step. The host learns the placement
only from periodic snapshots and never treats its own maps as the truth.

Ownership contract (a deliberate relaxation of the exclusive placement):
- A hot expert's previous RAM row keeps its bytes, recorded in
  `ram_shadow[row] = expert`. Evicting that expert later skips the D2H when
  its shadow row is still intact (all six tensors were written together, so
  one owner id covers them all). A shadow row is invalidated before it is
  reused as another expert's destination.
- VRAM rows: hot rows, the staging rows (rewritten every step, never
  promoted), and a free ring of F rows: a promotion takes the ring head and
  the victim's row takes its place, so the ring always holds F rows.
- RAM rows: the cold rows plus a pool of exactly R unreferenced rows, each
  tagged with the expert whose bytes it still holds (its shadow). A victim
  whose shadow is in the pool reclaims that row without a copy; otherwise
  the pool's round-robin head is invalidated and written. Either way the
  promoted expert's RAM row enters the pool as its shadow, so the pool size
  is invariant and RAM never grows.

This module holds the device state, the torch reference implementations
(the CPU tests and non-CUDA devices run them), and the Triton kernels for
the gather, the evict, and the flip. The planner lives in `device_lru.py`
(separate ownership); `reference_plan` here is the planner semantics the
runtime tests use until that module lands, and the contract both follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .staging import TENSORS


@dataclass
class PromoteTables:
    """Per-layer device state for promote mode; every tensor has a fixed address."""

    hot_map: Any  # [E] int32 expert -> logical hot slot / -1 (planner view)
    cold_map: Any  # [E] int32 expert -> logical cold slot / -1 (planner view)
    hot_rows: Any  # [hot_slots] int32 logical hot slot -> VRAM row
    cold_rows: Any  # [cold_slots] int32 logical cold slot -> RAM row
    hot_phys: Any  # [E] int32 expert -> VRAM row / -1 (kernel map)
    cold_phys: Any  # [E] int32 expert -> RAM row / -1 (kernel map)
    vram_free: Any  # [F] int32 ring of free VRAM rows (always full)
    ram_free: Any  # [R] int32 pool of unreferenced RAM rows (always full)
    ring_state: Any  # [2] int32: VRAM ring head, RAM pool round-robin head
    ram_shadow: Any  # [ram rows] int32 expert whose bytes the row holds / -1
    last_use: Any  # [E] int64 step of last selection
    clock: Any  # [1] int64
    error: Any  # [1] int32 sticky device error


@dataclass(frozen=True)
class StepPlan:
    """Planner output for one layer step; fixed shape S, `count` valid lanes."""

    promote_expert: Any  # [S] int64
    promote_cold_slot: Any  # [S] int64 logical
    victim_expert: Any  # [S] int64
    victim_hot_slot: Any  # [S] int64 logical
    count: Any  # [1] int32 promoted lanes
    staged_only_expert: Any  # [S] int64 misses that stay staging-only
    staged_only_cold_slot: Any  # [S] int64 logical
    staged_only_count: Any  # [1] int32


def allocate_tables(
    device, num_experts, hot_slots, cold_slots, vram_free_rows, ram_free_rows
):
    """Initial tables: logical slot i lives in physical row i; rings full."""
    import torch

    hot_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    cold_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    hot_map[:hot_slots] = torch.arange(hot_slots, dtype=torch.int32, device=device)
    cold_map[hot_slots:] = torch.arange(cold_slots, dtype=torch.int32, device=device)
    hot_rows = torch.arange(hot_slots, dtype=torch.int32, device=device)
    cold_rows = torch.arange(cold_slots, dtype=torch.int32, device=device)
    vram_free = torch.tensor(list(vram_free_rows), dtype=torch.int32, device=device)
    ram_free = torch.tensor(list(ram_free_rows), dtype=torch.int32, device=device)
    ring_state = torch.tensor([0, 0], dtype=torch.int32, device=device)
    ram_rows = cold_slots + len(ram_free_rows)
    ram_shadow = torch.full((ram_rows,), -1, dtype=torch.int32, device=device)
    ram_shadow[:cold_slots] = torch.arange(
        hot_slots, hot_slots + cold_slots, dtype=torch.int32, device=device
    )
    tables = PromoteTables(
        hot_map=hot_map,
        cold_map=cold_map,
        hot_rows=hot_rows,
        cold_rows=cold_rows,
        hot_phys=hot_map.clone(),
        cold_phys=cold_map.clone(),
        vram_free=vram_free,
        ram_free=ram_free,
        ring_state=ring_state,
        ram_shadow=ram_shadow,
        last_use=torch.zeros(num_experts, dtype=torch.int64, device=device),
        clock=torch.zeros(1, dtype=torch.int64, device=device),
        error=torch.zeros(1, dtype=torch.int32, device=device),
    )
    return tables


def capacity(tables):
    """Promotions one step may make: one free VRAM row each; RAM never limits."""
    return int(tables.vram_free.shape[0])


def reference_plan(ids, tables, staging_slots, enabled=True):
    """Planner semantics (torch, host-synchronizing; the device planner must match).

    Protect every valid selected expert (duplicates collapsed, hot hits
    included). Misses in ascending expert id become promotions up to
    min(distinct misses, unselected hot experts, free VRAM rows); victims are
    the unselected hot experts by (last_use, logical hot slot). The remaining
    misses are staged only. Invalid IDs never index and never plan. The
    planner owns recency: with the gate open it advances the clock and
    stamps every valid selected expert; with the gate closed (startup)
    nothing is promoted, recency is untouched, and every miss is staged only.
    """
    import torch

    num_experts = tables.hot_map.shape[0]
    selected = sorted(
        {int(e) for e in ids.reshape(-1).tolist() if 0 <= int(e) < num_experts}
    )
    hot_map = tables.hot_map.tolist()
    cold_map = tables.cold_map.tolist()
    misses = [e for e in selected if hot_map[e] < 0 and cold_map[e] >= 0]
    last_use = tables.last_use.tolist()
    victims = sorted(
        (e for e in range(num_experts) if hot_map[e] >= 0 and e not in selected),
        key=lambda e: (last_use[e], hot_map[e]),
    )
    limit = min(len(misses), len(victims), capacity(tables)) if enabled else 0
    promoted, staged = misses[:limit], misses[limit:]
    victims = victims[:limit]
    device = ids.device
    size = staging_slots
    if enabled:
        clock = int(tables.clock[0]) + 1
        for expert in selected:
            last_use[expert] = clock
        tables.last_use.copy_(torch.tensor(last_use, dtype=torch.int64, device=device))
        tables.clock.fill_(clock)

    def column(values):
        padded = list(values) + [0] * (size - len(values))
        return torch.tensor(padded[:size], dtype=torch.int64, device=device)

    return StepPlan(
        promote_expert=column(promoted),
        promote_cold_slot=column([cold_map[e] for e in promoted]),
        victim_expert=column(victims),
        victim_hot_slot=column([hot_map[e] for e in victims]),
        count=torch.tensor([limit], dtype=torch.int32, device=device),
        staged_only_expert=column(staged),
        staged_only_cold_slot=column([cold_map[e] for e in staged]),
        staged_only_count=torch.tensor([len(staged)], dtype=torch.int32, device=device),
    )


def apply_step_reference(tables, plan, staging_rows):
    """Reference for the flip kernel: returns (gather, evict, step_map) plans.

    Executed after the planner and before the copies, it decides the physical
    rows and updates the tables exactly as the device kernel does:

    - promotion i: destination VRAM row is the ring head; the victim's
      current VRAM row replaces it in the ring;
    - victim i: destination RAM row is the pool row still shadowing the
      victim (copy skipped) or else the pool's round-robin head (its old
      shadow invalidated, copy issued); the promoted expert's RAM row takes
      that pool position as the promoted expert's shadow;
    - the logical slot of the victim is taken over by the promoted expert,
      and the victim takes the promoted expert's logical cold slot;
    - staged-only misses map to the staging rows in order.
    Recency is the planner's; the flip never touches last_use or the clock.
    Returns row lists for the copies and the physical step map [E].
    """
    import torch

    count = int(plan.count[0])
    staged_count = int(plan.staged_only_count[0])
    hot_map = tables.hot_map.tolist()
    cold_map = tables.cold_map.tolist()
    hot_rows = tables.hot_rows.tolist()
    cold_rows = tables.cold_rows.tolist()
    hot_phys = tables.hot_phys.tolist()
    cold_phys = tables.cold_phys.tolist()
    vram_free = tables.vram_free.tolist()
    ram_free = tables.ram_free.tolist()
    vram_head, ram_head = tables.ring_state.tolist()
    shadow = tables.ram_shadow.tolist()
    gathers: list[tuple[int, int]] = []  # (ram row, vram row)
    evicts: list[tuple[int, int]] = []  # (vram row, ram row)
    if count > len(vram_free):
        raise RuntimeError("Promotion exceeds the free VRAM ring")
    for i in range(count):
        expert = int(plan.promote_expert[i])
        victim = int(plan.victim_expert[i])
        cold_slot = int(plan.promote_cold_slot[i])
        hot_slot = int(plan.victim_hot_slot[i])
        src_ram = cold_rows[cold_slot]
        victim_row = hot_rows[hot_slot]
        # VRAM: take the ring head, leave the victim's row in its place.
        dst_vram = vram_free[vram_head]
        vram_free[vram_head] = victim_row
        vram_head = (vram_head + 1) % len(vram_free)
        gathers.append((src_ram, dst_vram))
        # RAM: reclaim the victim's intact shadow, else write the pool head.
        position = next(
            (j for j, row in enumerate(ram_free) if shadow[row] == victim), -1
        )
        if position < 0:
            position = ram_head
            ram_head = (ram_head + 1) % len(ram_free)
            dst_ram = ram_free[position]
            shadow[dst_ram] = victim
            evicts.append((victim_row, dst_ram))
        else:
            dst_ram = ram_free[position]
        # The promoted expert's RAM row enters the pool as its shadow.
        ram_free[position] = src_ram
        shadow[src_ram] = expert
        # Flip the logical slots and physical rows.
        hot_rows[hot_slot] = dst_vram
        cold_rows[cold_slot] = dst_ram
        hot_map[expert], hot_map[victim] = hot_slot, -1
        cold_map[victim], cold_map[expert] = cold_slot, -1
        hot_phys[expert], hot_phys[victim] = dst_vram, -1
        cold_phys[victim], cold_phys[expert] = dst_ram, -1
    step_map = list(hot_phys)
    staged: list[tuple[int, int]] = []
    for i in range(staged_count):
        expert = int(plan.staged_only_expert[i])
        row = staging_rows[i]
        staged.append((cold_rows[int(plan.staged_only_cold_slot[i])], row))
        step_map[expert] = row
    device = tables.hot_map.device

    def write(target, values, dtype):
        target.copy_(torch.tensor(values, dtype=dtype, device=device))

    write(tables.hot_map, hot_map, torch.int32)
    write(tables.cold_map, cold_map, torch.int32)
    write(tables.hot_rows, hot_rows, torch.int32)
    write(tables.cold_rows, cold_rows, torch.int32)
    write(tables.hot_phys, hot_phys, torch.int32)
    write(tables.cold_phys, cold_phys, torch.int32)
    write(tables.vram_free, vram_free, torch.int32)
    write(tables.ram_free, ram_free, torch.int32)
    write(tables.ring_state, [vram_head, ram_head], torch.int32)
    write(tables.ram_shadow, shadow, torch.int32)
    return (
        gathers,
        staged,
        evicts,
        torch.tensor(step_map, dtype=torch.int32, device=device),
    )


def copy_rows_reference(source, destination, pairs):
    """Copy (src row, dst row) pairs for every bank tensor, in order."""
    for src, dst in pairs:
        for name in TENSORS:
            destination[name][dst].copy_(source[name][src])


def check_tables(tables, hot_slots, cold_slots):
    """Consistency of one layer's tables; raises on any violation.

    Every expert is in exactly one of hot/cold, logical slots are a
    permutation over their physical rows, physical maps agree with the
    logical maps through the row tables, and no VRAM row is both hot and
    free.
    """
    hot_map = tables.hot_map.tolist()
    cold_map = tables.cold_map.tolist()
    hot_rows = tables.hot_rows.tolist()
    cold_rows = tables.cold_rows.tolist()
    hot_phys = tables.hot_phys.tolist()
    cold_phys = tables.cold_phys.tolist()
    if sorted(v for v in hot_map if v >= 0) != list(range(hot_slots)):
        raise AssertionError("Every hot slot must have exactly one owner")
    if sorted(v for v in cold_map if v >= 0) != list(range(cold_slots)):
        raise AssertionError("Every cold slot must have exactly one owner")
    for expert, (h, c) in enumerate(zip(hot_map, cold_map)):
        if (h >= 0) == (c >= 0):
            raise AssertionError(f"Expert {expert} must be in exactly one partition")
        if hot_phys[expert] != (hot_rows[h] if h >= 0 else -1):
            raise AssertionError(f"Expert {expert}: hot row table disagrees")
        if cold_phys[expert] != (cold_rows[c] if c >= 0 else -1):
            raise AssertionError(f"Expert {expert}: cold row table disagrees")
    if len(set(hot_rows)) != len(hot_rows) or len(set(cold_rows)) != len(cold_rows):
        raise AssertionError("Physical rows must be unique per partition")
    vram_free = tables.vram_free.tolist()
    if len(set(vram_free)) != len(vram_free) or set(vram_free) & set(hot_rows):
        raise AssertionError("A VRAM row is both hot and free, or listed twice")
    ram_free = tables.ram_free.tolist()
    if len(set(ram_free)) != len(ram_free) or set(ram_free) & set(cold_rows):
        raise AssertionError("A RAM row is both cold and free, or listed twice")
    shadow = tables.ram_shadow.tolist()
    for c, row in enumerate(cold_rows):
        expert = cold_map.index(c)
        if shadow[row] != expert:
            raise AssertionError(f"Cold row {row} must shadow its expert {expert}")
    if int(tables.error[0]):
        raise RuntimeError("Promote mode recorded a device error")
