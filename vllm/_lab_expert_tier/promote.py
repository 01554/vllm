# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Promote mode: per-token cache management on the device, FreeToken style.

In promote mode the placement is owned by a device LRU planner instead of
the periodic heat policy. Every batch-1 decode step, inside the captured
graph and on the compute stream, each layer:

1. plans (planner module): which selected cold experts are promoted this
   step, which unselected hot experts are evicted to make room, and which
   remaining misses are only staged for this step;
2. flips the device tables so the promoted experts are hot and the victims
   are cold, and writes the copy lists and the step's expert map;
3. gathers the promoted and staged experts' rows from RAM into VRAM;
4. evicts the victims' rows from VRAM into RAM;
5. runs the MoE kernels on the bank through that map.

The tables are updated before the copies: they describe the placement the
rest of the step produces. All of it is queued on one stream with fixed
shapes, so ordering is by stream order alone: nothing reads the tables
between the flip and the MoE kernel, reads of a row precede the write that
recycles it, and no host synchronization is needed per step. The host reads
the tables only at forward boundaries (stats reports) and never treats its
own maps as the truth; a failure anywhere poisons the tier.

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

RAM backing (`backing=True`, opt-in) keeps every expert's RAM row for the
life of the process: RAM row `e` always holds expert `e`, so an eviction
only flips the tables and never writes back (FreeToken keeps the same
invariant). The RAM pool and shadows are unused; `ram_free` is a
placeholder of the VRAM ring's length so the planner's capacity stays
min(F, R) = F, and `cold_rows[slot]` is the evicted expert's own row.

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
    lru_state: Any = None  # planner-owned state (device_lru.allocate_state)
    backing: bool = False  # RAM row e always holds expert e; no write-back


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
    device,
    num_experts,
    hot_slots,
    cold_slots,
    vram_free_rows,
    ram_free_rows,
    backing=False,
):
    """Initial tables: logical slot i lives in physical row i; rings full.

    With `backing` the RAM bank has `num_experts` rows indexed by expert,
    logical cold slot i starts at row hot_slots + i, and `ram_free_rows`
    must be empty (the placeholder pool is sized like the VRAM ring).
    """
    import torch

    if backing and len(list(ram_free_rows)):
        raise ValueError("RAM backing has no RAM pool")

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
    if backing:
        cold_rows = cold_rows + hot_slots
        ram_free = torch.full_like(vram_free, -1)
        ram_shadow = torch.arange(num_experts, dtype=torch.int32, device=device)
    cold_phys = torch.where(cold_map >= 0, cold_rows[cold_map.clamp(min=0)], cold_map)
    tables = PromoteTables(
        hot_map=hot_map,
        cold_map=cold_map,
        hot_rows=hot_rows,
        cold_rows=cold_rows,
        hot_phys=hot_map.clone(),
        cold_phys=cold_phys,
        vram_free=vram_free,
        ram_free=ram_free,
        ring_state=ring_state,
        ram_shadow=ram_shadow,
        last_use=torch.zeros(num_experts, dtype=torch.int64, device=device),
        clock=torch.zeros(1, dtype=torch.int64, device=device),
        error=torch.zeros(1, dtype=torch.int32, device=device),
        backing=backing,
    )
    return tables


def capacity(tables):
    """Promotions one step may make: min(free VRAM ring, RAM pool) by contract.

    With RAM backing the pool is a placeholder of the ring's length, so the
    same expression yields the ring length.
    """
    return int(min(tables.vram_free.shape[0], tables.ram_free.shape[0]))


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
    With RAM backing the victim's destination is its own RAM row (its bytes
    are still there), nothing is evicted, and the pool and shadows are
    untouched.
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
    if count > len(vram_free) or count > len(ram_free):
        # The device flip assumes count <= min(F, R); a larger plan would
        # loop forever looking for a free pool position.
        raise RuntimeError("Promotion exceeds the free VRAM ring or RAM pool")
    # Pass 1: victims whose shadow is intact reclaim their pool position
    # first, so a later victim's write never invalidates a reclaimable one.
    positions = [-1] * count
    taken: set[int] = set()
    for i in range(count if not tables.backing else 0):
        victim = int(plan.victim_expert[i])
        for j, row in enumerate(ram_free):
            if j not in taken and shadow[row] == victim:
                positions[i], taken = j, taken | {j}
                break
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
        # RAM: pass 2 writes the round-robin head, skipping reclaimed
        # positions, after invalidating that row's old shadow.
        position = positions[i]
        if tables.backing:
            dst_ram = victim
        elif position < 0:
            while ram_head in taken:
                ram_head = (ram_head + 1) % len(ram_free)
            position = ram_head
            taken.add(position)
            ram_head = (ram_head + 1) % len(ram_free)
            dst_ram = ram_free[position]
            shadow[dst_ram] = victim
            evicts.append((victim_row, dst_ram))
        else:
            dst_ram = ram_free[position]
        if not tables.backing:
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
    shadow = tables.ram_shadow.tolist()
    if tables.backing:
        if ram_free != [-1] * len(vram_free):
            raise AssertionError("RAM backing keeps a placeholder pool only")
        if shadow != list(range(len(hot_map))):
            raise AssertionError("RAM backing keeps every expert in its own row")
        for c, row in enumerate(cold_rows):
            if row != cold_map.index(c):
                raise AssertionError(f"Cold slot {c} must point at its expert's row")
    else:
        if len(set(ram_free)) != len(ram_free) or set(ram_free) & set(cold_rows):
            raise AssertionError("A RAM row is both cold and free, or listed twice")
        for c, row in enumerate(cold_rows):
            expert = cold_map.index(c)
            if shadow[row] != expert:
                raise AssertionError(f"Cold row {row} must shadow its expert {expert}")
    if int(tables.error[0]):
        raise RuntimeError("Promote mode recorded a device error")


# ----------------------------------------------------------------------------
# Device execution: one flip program per layer step, one copy launch per
# direction. The torch references above define the semantics; CUDA runs these.
# ----------------------------------------------------------------------------

PLAN_WIDTH = 16
_KERNELS: dict[str, Any] = {}


@dataclass
class StepBuffers:
    """Fixed-address per-layer scratch the flip kernel fills every step."""

    gather_src: Any  # [2S] int32 RAM rows (promotions then staged-only)
    gather_dst: Any  # [2S] int32 VRAM rows
    gather_count: Any  # [1] int32
    evict_src: Any  # [S] int32 VRAM rows
    evict_dst: Any  # [S] int32 RAM rows
    evict_count: Any  # [1] int32
    evict_pos: Any  # [S] int32 flip scratch: reclaimed pool position or -1
    step_map: Any  # [E] int32 physical expert map for this step
    staging_rows: Any  # [S] int32 the layer's staging rows (constant)


def allocate_step_buffers(device, num_experts, width, staging_rows=()):
    """Allocate once per layer, before any capture: fixed addresses only."""
    import torch

    def ints(n):
        return torch.zeros(n, dtype=torch.int32, device=device)

    rows = list(staging_rows) + [0] * (width - len(staging_rows))
    return StepBuffers(
        gather_src=ints(2 * width),
        gather_dst=ints(2 * width),
        gather_count=ints(1),
        evict_src=ints(width),
        evict_dst=ints(width),
        evict_count=ints(1),
        evict_pos=ints(width),
        step_map=torch.full((num_experts,), -1, dtype=torch.int32, device=device),
        staging_rows=torch.tensor(rows[:width], dtype=torch.int32, device=device),
    )


def flip_step(tables, plan, buffers, staging_rows):
    """Apply the flip on the device (CUDA) or through the reference (else).

    Fills `buffers` with the copy lists and the step map. On CUDA this is a
    single Triton program; nothing touches the host.
    """
    device = tables.hot_map.device
    if device.type != "cuda":
        gathers, staged, evicts, step_map = apply_step_reference(
            tables, plan, list(staging_rows)
        )
        pairs = gathers + staged
        buffers.gather_count.fill_(len(pairs))
        buffers.evict_count.fill_(len(evicts))
        for i, (src, dst) in enumerate(pairs):
            buffers.gather_src[i], buffers.gather_dst[i] = src, dst
        for i, (src, dst) in enumerate(evicts):
            buffers.evict_src[i], buffers.evict_dst[i] = src, dst
        buffers.step_map.copy_(step_map)
        return
    # No host-to-device creation here: capture-safe, fixed addresses only.
    staging = buffers.staging_rows
    _flip_kernel()[(1,)](
        plan.promote_expert,
        plan.promote_cold_slot,
        plan.victim_expert,
        plan.victim_hot_slot,
        plan.count,
        plan.staged_only_expert,
        plan.staged_only_cold_slot,
        plan.staged_only_count,
        tables.hot_map,
        tables.cold_map,
        tables.hot_rows,
        tables.cold_rows,
        tables.hot_phys,
        tables.cold_phys,
        tables.vram_free,
        tables.ram_free,
        tables.ring_state,
        tables.ram_shadow,
        tables.error,
        staging,
        buffers.gather_src,
        buffers.gather_dst,
        buffers.gather_count,
        buffers.evict_src,
        buffers.evict_dst,
        buffers.evict_count,
        buffers.evict_pos,
        buffers.step_map,
        tables.hot_map.shape[0],
        tables.vram_free.shape[0],
        tables.ram_free.shape[0],
        WIDTH=PLAN_WIDTH,
        MAP_BLOCK=1024,
        BACKING=bool(tables.backing),
    )


COPY_PROGRAMS_PER_BANK = 32
COPY_WORDS = 4096  # int32 words (16 KiB) per program iteration


def copy_rows(source, destination, src_rows, dst_rows, count):
    """Copy `count` (src, dst) row pairs of every bank tensor; device count.

    One launch of a fixed small grid (banks x COPY_PROGRAMS_PER_BANK): each
    program streams its stripe of every row, reading `count` on the device
    (the FreeToken multi-bank copy shape), so an empty step costs a few
    programs instead of one per row chunk.
    """
    src_device = source[TENSORS[0]].device
    if src_device.type != "cuda":
        n = int(count.reshape(-1)[0].item())
        pairs = [(int(src_rows[i]), int(dst_rows[i])) for i in range(n)]
        copy_rows_reference(source, destination, pairs)
        return
    srcs = [_word_rows(source[name]) for name in TENSORS]
    dsts = [_word_rows(destination[name]) for name in TENSORS]
    for name, src, dst in zip(TENSORS, srcs, dsts):
        if src.shape[1] != dst.shape[1]:
            raise ValueError(f"{name}: destination row size differs from the source")
    grid = (len(TENSORS) * COPY_PROGRAMS_PER_BANK,)
    _copy_kernel()[grid](
        *srcs,
        *dsts,
        src_rows,
        dst_rows,
        count,
        *(dst.shape[1] for dst in dsts),
        *(src.stride(0) for src in srcs),
        *(dst.stride(0) for dst in dsts),
        PROGRAMS=COPY_PROGRAMS_PER_BANK,
        BLOCK=COPY_WORDS,
        num_warps=4,
    )


def _word_rows(tensor):
    """View a [rows, ...] contiguous tensor as [rows, int32 words]."""
    import torch

    if not tensor.is_contiguous():
        raise ValueError("Copies require contiguous bank rows")
    rows = tensor.shape[0]
    return tensor.view(torch.uint8).reshape(rows, -1).view(torch.int32)


def _byte_rows(tensor):
    import torch

    if not tensor.is_contiguous():
        raise ValueError("Promote copies require contiguous bank rows")
    return tensor.view(torch.uint8).reshape(tensor.shape[0], -1)


def _flip_kernel():
    """One program: the flip of `apply_step_reference`, on the device."""
    if "flip" in _KERNELS:
        return _KERNELS["flip"]
    from vllm.triton_utils import tl, triton

    @triton.jit
    def promote_flip(
        promote_expert_ptr,
        promote_cold_slot_ptr,
        victim_expert_ptr,
        victim_hot_slot_ptr,
        count_ptr,
        staged_expert_ptr,
        staged_cold_slot_ptr,
        staged_count_ptr,
        hot_map_ptr,
        cold_map_ptr,
        hot_rows_ptr,
        cold_rows_ptr,
        hot_phys_ptr,
        cold_phys_ptr,
        vram_free_ptr,
        ram_free_ptr,
        ring_state_ptr,
        ram_shadow_ptr,
        error_ptr,
        staging_ptr,
        gather_src_ptr,
        gather_dst_ptr,
        gather_count_ptr,
        evict_src_ptr,
        evict_dst_ptr,
        evict_count_ptr,
        evict_pos_ptr,
        step_map_ptr,
        num_experts,
        vram_ring,
        ram_pool,
        WIDTH: tl.constexpr,
        MAP_BLOCK: tl.constexpr,
        BACKING: tl.constexpr,
    ):
        count = tl.load(count_ptr)
        staged_count = tl.load(staged_count_ptr)
        vram_head = tl.load(ring_state_ptr)
        ram_head = tl.load(ring_state_ptr + 1)
        evicts = 0
        # Pass 1: record, per promotion lane, the pool position whose shadow
        # is the victim (or -1); a position claimed by one lane is not
        # matched again by another.
        if BACKING:  # noqa: SIM108 (constexpr branch inside the jit)
            pass_count = count * 0
        else:
            pass_count = count
        for i in range(0, pass_count):
            victim = tl.load(victim_expert_ptr + i)
            position = -1
            for j in range(0, ram_pool):
                row = tl.load(ram_free_ptr + j)
                owner = tl.load(ram_shadow_ptr + row)
                claimed = 0
                for k in range(0, i):
                    if tl.load(evict_pos_ptr + k) == j:
                        claimed = 1
                if (owner == victim) & (position < 0) & (claimed == 0):
                    position = j
            tl.store(evict_pos_ptr + i, position)
        tl.debug_barrier()
        for i in range(0, count):
            expert = tl.load(promote_expert_ptr + i)
            victim = tl.load(victim_expert_ptr + i)
            cold_slot = tl.load(promote_cold_slot_ptr + i)
            hot_slot = tl.load(victim_hot_slot_ptr + i)
            src_ram = tl.load(cold_rows_ptr + cold_slot)
            victim_row = tl.load(hot_rows_ptr + hot_slot)
            dst_vram = tl.load(vram_free_ptr + vram_head)
            tl.store(vram_free_ptr + vram_head, victim_row)
            vram_head = (vram_head + 1) % vram_ring
            tl.store(gather_src_ptr + i, src_ram)
            tl.store(gather_dst_ptr + i, dst_vram)
            position = tl.load(evict_pos_ptr + i)
            if BACKING:
                # The victim's own RAM row still holds its bytes.
                dst_ram = victim
            elif position < 0:
                # Round-robin head, skipping positions reclaimed in pass 1
                # or already handed out this step.
                claimed = 1
                while claimed == 1:
                    claimed = 0
                    for k in range(0, count):
                        if tl.load(evict_pos_ptr + k) == ram_head:
                            claimed = 1
                    if claimed == 1:
                        ram_head = (ram_head + 1) % ram_pool
                position = ram_head
                tl.store(evict_pos_ptr + i, position)
                ram_head = (ram_head + 1) % ram_pool
                dst_ram = tl.load(ram_free_ptr + position)
                tl.store(ram_shadow_ptr + dst_ram, victim)
                tl.store(evict_src_ptr + evicts, victim_row)
                tl.store(evict_dst_ptr + evicts, dst_ram)
                evicts += 1
            else:
                dst_ram = tl.load(ram_free_ptr + position)
            if not BACKING:
                tl.store(ram_free_ptr + position, src_ram)
                tl.store(ram_shadow_ptr + src_ram, expert)
            tl.store(hot_rows_ptr + hot_slot, dst_vram)
            tl.store(cold_rows_ptr + cold_slot, dst_ram)
            tl.store(hot_map_ptr + expert, hot_slot)
            tl.store(hot_map_ptr + victim, -1)
            tl.store(cold_map_ptr + victim, cold_slot)
            tl.store(cold_map_ptr + expert, -1)
            tl.store(hot_phys_ptr + expert, dst_vram)
            tl.store(hot_phys_ptr + victim, -1)
            tl.store(cold_phys_ptr + victim, dst_ram)
            tl.store(cold_phys_ptr + expert, -1)
        tl.store(ring_state_ptr, vram_head)
        tl.store(ring_state_ptr + 1, ram_head)
        tl.store(evict_count_ptr, evicts)
        tl.debug_barrier()
        # Step map: the physical hot map with staged-only misses overlaid.
        for start in range(0, num_experts, MAP_BLOCK):
            offs = start + tl.arange(0, MAP_BLOCK)
            in_range = offs < num_experts
            rows = tl.load(hot_phys_ptr + offs, mask=in_range, other=-1)
            tl.store(step_map_ptr + offs, rows, mask=in_range)
        tl.debug_barrier()
        for i in range(0, staged_count):
            expert = tl.load(staged_expert_ptr + i)
            cold_slot = tl.load(staged_cold_slot_ptr + i)
            row = tl.load(staging_ptr + i)
            tl.store(gather_src_ptr + count + i, tl.load(cold_rows_ptr + cold_slot))
            tl.store(gather_dst_ptr + count + i, row)
            tl.store(step_map_ptr + expert, row)
        tl.store(gather_count_ptr, count + staged_count)

    _KERNELS["flip"] = promote_flip
    return promote_flip


def _copy_kernel():
    """Fixed grid: program (bank, stripe) streams its columns of every row."""
    if "copy" in _KERNELS:
        return _KERNELS["copy"]
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _stripe(
        src,
        dst,
        src_rows_ptr,
        dst_rows_ptr,
        count,
        words,
        sstride,
        dstride,
        stripe,
        PROGRAMS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        for lane in range(0, count):
            src_row = tl.load(src_rows_ptr + lane).to(tl.int64)
            dst_row = tl.load(dst_rows_ptr + lane).to(tl.int64)
            src_base = src + src_row * sstride
            dst_base = dst + dst_row * dstride
            for start in range(stripe * BLOCK, words, PROGRAMS * BLOCK):
                offsets = start + tl.arange(0, BLOCK)
                mask = offsets < words
                values = tl.load(src_base + offsets, mask=mask)
                tl.store(dst_base + offsets, values, mask=mask)

    @triton.jit
    def promote_copy(
        src0,
        src1,
        src2,
        src3,
        src4,
        src5,
        dst0,
        dst1,
        dst2,
        dst3,
        dst4,
        dst5,
        src_rows_ptr,
        dst_rows_ptr,
        count_ptr,
        words0,
        words1,
        words2,
        words3,
        words4,
        words5,
        sstride0,
        sstride1,
        sstride2,
        sstride3,
        sstride4,
        sstride5,
        dstride0,
        dstride1,
        dstride2,
        dstride3,
        dstride4,
        dstride5,
        PROGRAMS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        which = tl.program_id(0) // PROGRAMS
        stripe = tl.program_id(0) % PROGRAMS
        count = tl.load(count_ptr)
        if which == 0:
            _stripe(
                src0,
                dst0,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words0,
                sstride0,
                dstride0,
                stripe,
                PROGRAMS,
                BLOCK,
            )
        elif which == 1:
            _stripe(
                src1,
                dst1,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words1,
                sstride1,
                dstride1,
                stripe,
                PROGRAMS,
                BLOCK,
            )
        elif which == 2:
            _stripe(
                src2,
                dst2,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words2,
                sstride2,
                dstride2,
                stripe,
                PROGRAMS,
                BLOCK,
            )
        elif which == 3:
            _stripe(
                src3,
                dst3,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words3,
                sstride3,
                dstride3,
                stripe,
                PROGRAMS,
                BLOCK,
            )
        elif which == 4:
            _stripe(
                src4,
                dst4,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words4,
                sstride4,
                dstride4,
                stripe,
                PROGRAMS,
                BLOCK,
            )
        else:
            _stripe(
                src5,
                dst5,
                src_rows_ptr,
                dst_rows_ptr,
                count,
                words5,
                sstride5,
                dstride5,
                stripe,
                PROGRAMS,
                BLOCK,
            )

    _KERNELS["copy"] = promote_copy
    return promote_copy
