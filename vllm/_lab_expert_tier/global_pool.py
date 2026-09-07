# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Global expert pool: one VRAM bank shared by every layer, FreeToken style.

Promote mode keeps a fixed number of hot rows per layer. FreeToken keeps
one cache of rows for the whole model and evicts the least recently used
expert of any layer, so busy layers can hold more experts than quiet ones.
This module is that pool on top of RAM backing (every expert keeps its RAM
row, so an eviction never copies out):

- One bank of `pool_rows + staging` rows per tensor name. Row `r` holds
  the expert whose *key* is `row_key[r]` (key = layer * E + expert), or
  is free (-1). `hot_phys[key]` is the row or -1; its per-layer slices are
  the kernel maps each layer reads. `cold_phys[key]` is the expert's RAM
  row (= expert id under backing) while it is not resident, else -1, so
  the eager two-partition path reads the same maps.
- Each layer's batch-1 step is one program: distinct valid selections in
  first-occurrence order are stamped with the step clock; each miss takes
  the row of the least recently used resident expert of any layer (ties
  by key), copying in from this layer's RAM bank; a miss that finds no
  victim (or any miss while the gate is closed) is staged only into the
  shared staging rows for this step. The step map is this layer's slice
  of `hot_phys` with the staged experts overlaid.
- Everything runs on the compute stream with fixed shapes and addresses;
  the host reads the tables only at stats reports, where they are
  validated.

The torch reference below defines the semantics; CPU tests and non-CUDA
devices run it, and the Triton program must match it exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .staging import TENSORS

PLAN_WIDTH = 16
KEY_BITS = 16  # keys < 2**16: 48 layers x 512 experts = 24576


@dataclass
class GlobalTables:
    """Pool-wide device state; every tensor has a fixed address."""

    num_layers: int
    num_experts: int
    pool_rows: int
    hot_phys: Any  # [L*E] int32 key -> bank row / -1
    cold_phys: Any  # [L*E] int32 key -> RAM row (expert id) / -1
    row_key: Any  # [pool_rows + staging] int32 row -> key / -1
    last_use: Any  # [L*E] int64 step of last selection
    clock: Any  # [1] int64
    gate: Any  # [1] int32 promotions allowed
    error: Any  # [1] int32 sticky device error
    staging_rows: Any  # [S] int32 shared staging rows (constant)

    @property
    def keys(self):
        return self.num_layers * self.num_experts

    def layer_slice(self, table, layer):
        start = layer * self.num_experts
        return table[start : start + self.num_experts]


@dataclass
class StepBuffers:
    """Per-layer fixed-address scratch the step program fills."""

    gather_src: Any  # [W] int32 RAM rows
    gather_dst: Any  # [W] int32 bank rows
    gather_count: Any  # [1] int32
    staged_expert: Any  # [W] int32 (scratch for the map overlay)
    staged_row: Any  # [W] int32
    staged_count: Any  # [1] int32
    promoted_count: Any  # [1] int32
    step_map: Any  # [E] int32


def allocate_global_tables(device, num_experts, slots_per_layer, staging):
    """Layer l's first `slots_per_layer[l]` experts start resident, packed
    in layer order; the `staging` rows follow the pool and stay free."""
    import torch

    num_layers = len(slots_per_layer)
    if staging < 1 or any(not 0 < s < num_experts for s in slots_per_layer):
        raise ValueError("Global pool needs staging rows and partial layers")
    if num_layers * num_experts >= 1 << KEY_BITS:
        raise ValueError("Global pool keys exceed the packed key width")
    keys = num_layers * num_experts
    pool_rows = sum(slots_per_layer)
    hot_phys = torch.full((keys,), -1, dtype=torch.int32)
    cold_phys = torch.arange(num_experts, dtype=torch.int32).repeat(num_layers)
    row_key = torch.full((pool_rows + staging,), -1, dtype=torch.int32)
    offset = 0
    for layer, slots in enumerate(slots_per_layer):
        base = layer * num_experts
        hot_phys[base : base + slots] = torch.arange(
            offset, offset + slots, dtype=torch.int32
        )
        cold_phys[base : base + slots] = -1
        row_key[offset : offset + slots] = torch.arange(
            base, base + slots, dtype=torch.int32
        )
        offset += slots
    return GlobalTables(
        num_layers=num_layers,
        num_experts=num_experts,
        pool_rows=pool_rows,
        hot_phys=hot_phys.to(device),
        cold_phys=cold_phys.to(device),
        row_key=row_key.to(device),
        last_use=torch.zeros(keys, dtype=torch.int64, device=device),
        clock=torch.zeros(1, dtype=torch.int64, device=device),
        gate=torch.zeros(1, dtype=torch.int32, device=device),
        error=torch.zeros(1, dtype=torch.int32, device=device),
        staging_rows=torch.arange(
            pool_rows, pool_rows + staging, dtype=torch.int32, device=device
        ),
    )


def allocate_step_buffers(device, num_experts, width=PLAN_WIDTH):
    import torch

    def ints(n):
        return torch.zeros(n, dtype=torch.int32, device=device)

    return StepBuffers(
        gather_src=ints(width),
        gather_dst=ints(width),
        gather_count=ints(1),
        staged_expert=ints(width),
        staged_row=ints(width),
        staged_count=ints(1),
        promoted_count=ints(1),
        step_map=torch.full((num_experts,), -1, dtype=torch.int32, device=device),
    )


def set_gate(tables, enabled):
    tables.gate.fill_(1 if enabled else 0)


def step_reference(tables, layer, ids, buffers):
    """Plan and flip one layer step on the host (torch, synchronizing).

    Returns (gathers, step_map) with gathers as (RAM row, bank row) pairs in
    copy order: promotions first, then staged-only misses. Semantics:

    - `ids` values outside [0, E) other than -1 set the sticky error and
      are skipped; -1 is padding.
    - Distinct valid selections in first-occurrence order. With the gate
      open the clock advances and every selection is stamped.
    - Each miss, in that order, evicts the resident key (any layer) with
      the smallest (last_use, key) among keys not stamped this step and
      takes its row; without such a key it is staged only. With the gate
      closed every miss is staged only and recency is untouched.
    """
    import torch

    E = tables.num_experts
    if not 0 <= layer < tables.num_layers:
        raise ValueError("Layer index outside the pool")
    hot = tables.hot_phys.tolist()
    cold = tables.cold_phys.tolist()
    row_key = tables.row_key.tolist()
    last_use = tables.last_use.tolist()
    staging_rows = tables.staging_rows.tolist()
    gate = bool(int(tables.gate[0]))
    raw = [int(v) for v in ids.reshape(-1).tolist()]
    if len(raw) > buffers.gather_src.shape[0] or len(raw) > len(staging_rows):
        raise ValueError("Step ids exceed the plan width or the staging rows")
    error = bool(int(tables.error[0]))
    selected: list[int] = []
    for value in raw:
        if value == -1:
            continue
        if not 0 <= value < E:
            error = True
            continue
        if value not in selected:
            selected.append(value)
    clock = int(tables.clock[0])
    if gate:
        clock += 1
        for e in selected:
            last_use[layer * E + e] = clock
    base = layer * E
    gathers: list[tuple[int, int]] = []
    staged: list[tuple[int, int]] = []
    for e in selected:
        key = base + e
        if hot[key] >= 0:
            continue
        victim = -1
        if gate:
            best = None
            for k in range(tables.keys):
                if hot[k] >= 0 and last_use[k] < clock:
                    candidate = (last_use[k], k)
                    if best is None or candidate < best:
                        best = candidate
            if best is not None:
                victim = best[1]
        if victim < 0:
            staged.append((e, staging_rows[len(staged)]))
            continue
        row = hot[victim]
        hot[victim], cold[victim] = -1, victim % E
        hot[key], cold[key] = row, -1
        row_key[row] = key
        gathers.append((e, row))
    step_map = hot[base : base + E]
    for e, row in staged:
        step_map[e] = row
    device = tables.hot_phys.device

    def write(target, values, dtype):
        target.copy_(torch.tensor(values, dtype=dtype, device=device))

    write(tables.hot_phys, hot, torch.int32)
    write(tables.cold_phys, cold, torch.int32)
    write(tables.row_key, row_key, torch.int32)
    write(tables.last_use, last_use, torch.int64)
    tables.clock.fill_(clock)
    tables.error.fill_(1 if error else 0)
    pairs = gathers + [(e, row) for e, row in staged]
    buffers.gather_count.fill_(len(pairs))
    buffers.promoted_count.fill_(len(gathers))
    buffers.staged_count.fill_(len(staged))
    for i, (src, dst) in enumerate(pairs):
        buffers.gather_src[i], buffers.gather_dst[i] = src, dst
    for i, (e, row) in enumerate(staged):
        buffers.staged_expert[i], buffers.staged_row[i] = e, row
    write(buffers.step_map, step_map, torch.int32)
    return pairs, buffers.step_map


def step(tables, layer, ids, buffers):
    """Plan and flip one layer step: Triton on CUDA, the reference elsewhere."""
    if tables.hot_phys.device.type != "cuda":
        step_reference(tables, layer, ids, buffers)
        return
    flat = ids.reshape(-1)
    if not flat.is_contiguous():
        raise ValueError("Global step requires contiguous ids")
    width = buffers.gather_src.shape[0]
    if flat.numel() > width or flat.numel() > tables.staging_rows.shape[0]:
        raise ValueError("Step ids exceed the plan width or the staging rows")
    keys = tables.keys
    block = 1024
    _step_kernel()[(1,)](
        flat,
        flat.numel(),
        layer,
        tables.hot_phys,
        tables.cold_phys,
        tables.row_key,
        tables.last_use,
        tables.clock,
        tables.gate,
        tables.error,
        tables.staging_rows,
        buffers.gather_src,
        buffers.gather_dst,
        buffers.gather_count,
        buffers.staged_expert,
        buffers.staged_row,
        buffers.staged_count,
        buffers.promoted_count,
        buffers.step_map,
        tables.num_experts,
        keys,
        WIDTH=width,
        BLOCK=block,
        NUM_BLOCKS=(keys + block - 1) // block,
        MAP_BLOCK=1024,
    )


def check_global_tables(tables):
    """Consistency of the pool; raises on any violation.

    Every key is resident or has its RAM row, never both; resident keys and
    rows are a bijection; staging rows are never owned; no device error.
    """
    E = tables.num_experts
    hot = tables.hot_phys.tolist()
    cold = tables.cold_phys.tolist()
    row_key = tables.row_key.tolist()
    staging = set(tables.staging_rows.tolist())
    owners = {}
    for key, (h, c) in enumerate(zip(hot, cold)):
        if (h >= 0) == (c >= 0):
            raise AssertionError(f"Key {key} must be resident or backed, not both")
        if c >= 0 and c != key % E:
            raise AssertionError(f"Key {key} must be backed by its own RAM row")
        if h >= 0:
            if h in staging or not 0 <= h < tables.pool_rows:
                raise AssertionError(f"Key {key} owns a row outside the pool")
            if h in owners:
                raise AssertionError(f"Row {h} has two owners")
            owners[h] = key
    for row, key in enumerate(row_key):
        if owners.get(row, -1) != key:
            raise AssertionError(f"Row {row} owner table disagrees")
    if int(tables.error[0]):
        raise RuntimeError("Global pool recorded a device error")


def resident_per_layer(tables):
    hot = tables.hot_phys.view(tables.num_layers, tables.num_experts)
    return (hot >= 0).sum(dim=1).tolist()


_KERNELS: dict[str, Any] = {}


def _step_kernel():
    """One program: `step_reference` on the device."""
    if "step" in _KERNELS:
        return _KERNELS["step"]
    from vllm.triton_utils import tl, triton

    @triton.jit
    def global_pool_step(
        ids_ptr,
        n,
        layer,
        hot_phys_ptr,
        cold_phys_ptr,
        row_key_ptr,
        last_use_ptr,
        clock_ptr,
        gate_ptr,
        error_ptr,
        staging_ptr,
        gather_src_ptr,
        gather_dst_ptr,
        gather_count_ptr,
        staged_expert_ptr,
        staged_row_ptr,
        staged_count_ptr,
        promoted_count_ptr,
        step_map_ptr,
        num_experts,
        num_keys,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        MAP_BLOCK: tl.constexpr,
    ):
        key_max = 0x7FFFFFFFFFFFFFFF
        lane = tl.arange(0, WIDTH)
        present = lane < n
        raw = tl.load(ids_ptr + lane, mask=present, other=-1).to(tl.int64)
        valid = present & (raw >= 0) & (raw < num_experts)
        bad = present & (raw != -1) & (~valid)
        if tl.sum(bad.to(tl.int32), 0) > 0:
            tl.store(error_ptr, 1)
        safe = tl.where(valid, raw, 0)
        same = safe[:, None] == safe[None, :]
        earlier = lane[None, :] < lane[:, None]
        duplicate = tl.sum((same & earlier & valid[None, :]).to(tl.int32), 1) > 0
        distinct = valid & (duplicate == 0)
        base = layer.to(tl.int64) * num_experts
        keys = base + safe
        gate = tl.load(gate_ptr) != 0
        clock = tl.load(clock_ptr)
        if gate:
            clock = clock + 1
            tl.store(clock_ptr, clock)
            tl.store(last_use_ptr + keys, clock, mask=distinct)
        tl.debug_barrier()
        promoted = 0
        staged = 0
        for i in range(0, WIDTH):
            is_distinct = tl.sum(tl.where(lane == i, distinct.to(tl.int32), 0), 0)
            if is_distinct > 0:
                expert = tl.load(ids_ptr + i).to(tl.int64)
                key = base + expert
                resident = tl.load(hot_phys_ptr + key)
                if resident < 0:
                    best = tl.full((), key_max, tl.int64)
                    if gate:
                        for block in range(0, NUM_BLOCKS):
                            offs = block * BLOCK + tl.arange(0, BLOCK)
                            in_range = offs < num_keys
                            rows = tl.load(hot_phys_ptr + offs, mask=in_range, other=-1)
                            use = tl.load(
                                last_use_ptr + offs, mask=in_range, other=key_max
                            )
                            candidate = in_range & (rows >= 0) & (use < clock)
                            packed = tl.where(
                                candidate,
                                use * (1 << 16) + offs.to(tl.int64),
                                key_max,
                            )
                            best = tl.minimum(best, tl.min(packed, 0))
                    if best != key_max:
                        victim = best % (1 << 16)
                        row = tl.load(hot_phys_ptr + victim)
                        tl.store(hot_phys_ptr + victim, -1)
                        tl.store(cold_phys_ptr + victim, (victim % num_experts))
                        tl.store(hot_phys_ptr + key, row)
                        tl.store(cold_phys_ptr + key, -1)
                        tl.store(row_key_ptr + row, key.to(tl.int32))
                        tl.store(gather_src_ptr + promoted, expert.to(tl.int32))
                        tl.store(gather_dst_ptr + promoted, row)
                        promoted += 1
                    else:
                        tl.store(staged_expert_ptr + staged, expert.to(tl.int32))
                        tl.store(staged_row_ptr + staged, tl.load(staging_ptr + staged))
                        staged += 1
        tl.store(promoted_count_ptr, promoted)
        tl.store(staged_count_ptr, staged)
        tl.store(gather_count_ptr, promoted + staged)
        tl.debug_barrier()
        for i in range(0, staged):
            tl.store(gather_src_ptr + promoted + i, tl.load(staged_expert_ptr + i))
            tl.store(gather_dst_ptr + promoted + i, tl.load(staged_row_ptr + i))
        for start in range(0, num_experts, MAP_BLOCK):
            offs = start + tl.arange(0, MAP_BLOCK)
            in_range = offs < num_experts
            rows = tl.load(hot_phys_ptr + base + offs, mask=in_range, other=-1)
            tl.store(step_map_ptr + offs, rows, mask=in_range)
        tl.debug_barrier()
        for i in range(0, staged):
            expert = tl.load(staged_expert_ptr + i)
            tl.store(step_map_ptr + expert, tl.load(staged_row_ptr + i))

    _KERNELS["step"] = global_pool_step
    return global_pool_step


class GlobalPool:
    """The shared bank, its staging views, the tables, and the layer offsets."""

    def __init__(self, device, sources, slots_per_layer, staging):
        import torch

        self.slots_per_layer = list(slots_per_layer)
        self.staging_slots = staging
        self.tables = allocate_global_tables(
            device, sources[TENSORS[0]].shape[0], self.slots_per_layer, staging
        )
        self.rows = self.tables.pool_rows + staging
        self.offsets = [0]
        for slots in self.slots_per_layer[:-1]:
            self.offsets.append(self.offsets[-1] + slots)
        self.bank = {
            name: torch.zeros(
                (self.rows, *source.shape[1:]), dtype=source.dtype, device=device
            )
            for name, source in sources.items()
        }
        self.staging = {
            name: tensor[self.tables.pool_rows :] for name, tensor in self.bank.items()
        }
        self.row_bytes = sum(t[0].numel() * t.element_size() for t in sources.values())
        self.staging_bytes = self.row_bytes * staging
        self.pool_bytes = self.row_bytes * self.tables.pool_rows

    def offset(self, layer):
        return self.offsets[layer]

    def host_swap(self, layer, old_expert, new_expert):
        """Gate-closed exchange for init verification: `new_expert` takes the
        row of resident `old_expert`, which falls back to its RAM row. The
        caller copies the bytes."""
        tables = self.tables
        if int(tables.gate[0]):
            raise RuntimeError("Host swaps are only allowed while the gate is closed")
        E = tables.num_experts
        old_key, new_key = layer * E + old_expert, layer * E + new_expert
        row = int(tables.hot_phys[old_key])
        if row < 0 or int(tables.hot_phys[new_key]) >= 0:
            raise AssertionError("Swap does not match the current pool placement")
        tables.hot_phys[old_key], tables.cold_phys[old_key] = -1, old_expert
        tables.hot_phys[new_key], tables.cold_phys[new_key] = row, -1
        tables.row_key[row] = new_key

    def snapshot(self):
        """Validate the pool on the host; one copy per stats report."""
        check_global_tables(self.tables)
        return resident_per_layer(self.tables)


def copy_in(source, bank, buffers):
    """Copy the planned RAM rows of this layer into the bank rows."""
    from .promote import copy_rows

    copy_rows(
        source, bank, buffers.gather_src, buffers.gather_dst, buffers.gather_count
    )


__all__ = [
    "TENSORS",
    "GlobalPool",
    "GlobalTables",
    "StepBuffers",
    "allocate_global_tables",
    "allocate_step_buffers",
    "check_global_tables",
    "copy_in",
    "resident_per_layer",
    "set_gate",
    "step",
    "step_reference",
]
