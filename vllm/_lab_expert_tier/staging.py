# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-time VRAM staging of the cold experts one forward selects.

The exclusive placement stays as it is: hot experts live in VRAM, cold
experts live in pinned RAM and the RAM copy remains the owner. For a batch-1
decode forward, at most `top_k` distinct cold experts can be selected per
layer, so each layer keeps `top_k` spare rows at the end of its hot bank.
Before the layer's MoE kernel runs, the selected cold rows are copied into
those spare rows and the step's expert map points the selected cold experts
at them; hot experts and staged experts then run through one Marlin chain
from VRAM instead of the cold partition reading RAM through UVA.

Everything here has fixed shapes and issues no host synchronization, so the
sequence can be captured in a CUDA Graph and replayed with different routing.
On CUDA each layer costs two launches:

- `plan_staging` is one Triton program that derives, from this step's
  routing IDs and the current cold and hot maps, the distinct selected cold
  experts (invalid and padding IDs ignored, duplicates collapsed, staging
  slots assigned in first-occurrence order), a gather index per slot, the
  device count, and the step's expert map, rebuilt from the hot map every
  step so a previous step's assignment can never leak into the next.
- `gather_staging` is one Triton launch that copies exactly `count` rows of
  all six bank tensors from the cold source to the staging rows, reading the
  count on the device and exiting early for unused slots, so a zero-miss
  step copies nothing.

The first version issued the plan as about thirty torch ops per layer and
six copy launches, and measured 7.7% slower than the fused two-partition
path on the RTX PRO 6000 (B 30.49 vs 33.05 tok/s) with the resident hit
rate down from 91.8% to 88.8% because ten hot slots per layer became
staging rows. Which share of the loss is launch volume and which is the
lower hit rate has not been separated; this version removes the launch
volume so the next measurement isolates the hit-rate cost. The torch
implementation remains the reference for CPU tests and non-CUDA devices
and defines the semantics the kernels must match.

The row layout is the Marlin-packed layout the banks already use; nothing is
re-packed. Staged rows are transient copies and are never written back.
"""

from __future__ import annotations

from typing import Any

TENSORS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
)
# Routing width the fused plan kernel is compiled for; ids are padded to it.
PLAN_WIDTH = 16
COPY_BLOCK = 4096


def plan_staging(ids, cold_map, hot_map, hot_slots, staging_slots):
    """Fixed-shape staging plan for one forward.

    Args:
        ids: [rows, k] routing IDs; -1 marks padding. rows * k must not
            exceed staging_slots, and rows * k must not exceed PLAN_WIDTH.
        cold_map: [E] int32 device map expert -> cold RAM slot or -1.
        hot_map: [E] int32 device map expert -> hot VRAM slot or -1.
        hot_slots: number of resident hot rows; staging rows follow them.
        staging_slots: spare rows at the end of the hot bank.

    Returns:
        gather_index: [staging_slots] int64 cold slots to copy, in slot
            order; entries at or past `count` are unused and hold 0.
        expert_map: [E] int32 this step's map: hot experts keep their hot
            slot, staged cold experts map to hot_slots + staging slot in
            first-occurrence order, and every other expert maps to -1.
        count: [1] int32 device tensor, the number of staged experts.
    """
    flat = ids.reshape(-1)
    if flat.shape[0] > staging_slots:
        raise ValueError("Forward selects more slots than the staging rows")
    if flat.shape[0] > PLAN_WIDTH:
        raise ValueError("Forward is wider than the staging plan kernel")
    if ids.device.type == "cuda":
        return _plan_staging_cuda(flat, cold_map, hot_map, hot_slots, staging_slots)
    return plan_staging_reference(flat, cold_map, hot_map, hot_slots, staging_slots)


def plan_staging_reference(flat, cold_map, hot_map, hot_slots, staging_slots):
    """Torch implementation; defines the semantics of the fused kernel."""
    import torch

    num_experts = hot_map.shape[0]
    n = flat.shape[0]
    valid = (flat >= 0) & (flat < num_experts)
    # Clamp before any gather so an invalid ID can never index out of range;
    # its result is masked out below.
    safe = torch.where(valid, flat, torch.zeros_like(flat)).to(torch.int64)
    cold_slot = cold_map[safe].to(torch.int64)
    is_cold = valid & (cold_slot >= 0)
    # An element is distinct when no earlier cold element carries its ID.
    same = safe[:, None] == safe[None, :]
    earlier = (
        torch.arange(n, device=flat.device)[None, :]
        < torch.arange(n, device=flat.device)[:, None]
    )
    duplicate = (same & earlier & is_cold[None, :]).any(dim=1)
    distinct = is_cold & ~duplicate
    slot = torch.cumsum(distinct.to(torch.int64), 0) - 1
    count = distinct.sum().to(torch.int32).reshape(1)
    # Scatter through one spare "dummy" index so non-distinct positions land
    # nowhere visible; that keeps every shape static.
    gather = torch.zeros(staging_slots + 1, dtype=torch.int64, device=flat.device)
    gather.scatter_(
        0, torch.where(distinct, slot, torch.full_like(slot, staging_slots)), cold_slot
    )
    expert_map = torch.full(
        (num_experts + 1,), -1, dtype=torch.int32, device=flat.device
    )
    expert_map[:num_experts].copy_(hot_map)
    expert_map.scatter_(
        0,
        torch.where(distinct, safe, torch.full_like(safe, num_experts)),
        (hot_slots + slot).to(torch.int32),
    )
    return gather[:staging_slots], expert_map[:num_experts], count


def _plan_staging_cuda(flat, cold_map, hot_map, hot_slots, staging_slots):
    import torch

    num_experts = hot_map.shape[0]
    gather = torch.zeros(staging_slots, dtype=torch.int64, device=flat.device)
    expert_map = torch.empty(num_experts, dtype=torch.int32, device=flat.device)
    count = torch.zeros(1, dtype=torch.int32, device=flat.device)
    _plan_kernel()[(1,)](
        flat,
        cold_map,
        hot_map,
        gather,
        expert_map,
        count,
        flat.shape[0],
        num_experts,
        hot_slots,
        WIDTH=PLAN_WIDTH,
        MAP_BLOCK=1024,
    )
    return gather, expert_map, count


def gather_staging_reference(source, staging, gather_index, count):
    """Copy `count` rows of every tensor; host-synchronizing, for tests."""
    rows = int(count.reshape(-1)[0].item())
    for name in TENSORS:
        for slot in range(rows):
            staging[name][slot].copy_(source[name][gather_index[slot]])


def gather_staging(source, staging, gather_index, count):
    """Copy the planned cold rows into the staging rows of every bank tensor.

    `source[name]` is the cold bank as a device-visible tensor (UVA view of
    pinned RAM) and `staging[name]` the [staging_slots, ...] view at the end
    of the hot bank. On CUDA one launch covers all six tensors, reads
    `count` on the device, and never synchronizes; elsewhere the reference
    path is used.
    """
    if source[TENSORS[0]].device.type != "cuda":
        gather_staging_reference(source, staging, gather_index, count)
        return
    srcs = [_byte_rows(source[name]) for name in TENSORS]
    dsts = [_byte_rows(staging[name]) for name in TENSORS]
    for name, src, dst in zip(TENSORS, srcs, dsts):
        if src.shape[1] != dst.shape[1]:
            raise ValueError(f"{name}: staging row size differs from the source")
    widest = max(dst.shape[1] for dst in dsts)
    grid = (dsts[0].shape[0], (widest + COPY_BLOCK - 1) // COPY_BLOCK, len(TENSORS))
    _copy_kernel()[grid](
        *srcs,
        *dsts,
        gather_index,
        count,
        *(dst.shape[1] for dst in dsts),
        *(src.stride(0) for src in srcs),
        *(dst.stride(0) for dst in dsts),
        BLOCK=COPY_BLOCK,
    )


def _byte_rows(tensor):
    """View a [rows, ...] contiguous tensor as [rows, bytes] uint8."""
    import torch

    if not tensor.is_contiguous():
        raise ValueError("Staging requires contiguous bank rows")
    rows = tensor.shape[0]
    return tensor.view(torch.uint8).reshape(rows, -1)


_KERNELS: dict[str, Any] = {}


def _plan_kernel():
    """One program: distinct cold selections, slots, gather index, map."""
    if "plan" in _KERNELS:
        return _KERNELS["plan"]
    from vllm.triton_utils import tl, triton

    @triton.jit
    def staging_plan(
        ids_ptr,
        cold_map_ptr,
        hot_map_ptr,
        gather_ptr,
        map_ptr,
        count_ptr,
        n,
        num_experts,
        hot_slots,
        WIDTH: tl.constexpr,
        MAP_BLOCK: tl.constexpr,
    ):
        lane = tl.arange(0, WIDTH)
        present = lane < n
        ids = tl.load(ids_ptr + lane, mask=present, other=-1).to(tl.int64)
        valid = present & (ids >= 0) & (ids < num_experts)
        safe = tl.where(valid, ids, 0)
        cold = tl.load(cold_map_ptr + safe, mask=valid, other=-1).to(tl.int64)
        is_cold = valid & (cold >= 0)
        # Distinct: no earlier cold lane carries the same expert ID.
        same = safe[:, None] == safe[None, :]
        earlier = lane[None, :] < lane[:, None]
        duplicate = tl.sum((same & earlier & is_cold[None, :]).to(tl.int32), 1) > 0
        distinct = is_cold & (duplicate == 0)
        ones = distinct.to(tl.int64)
        slot = tl.cumsum(ones, 0) - 1
        count = tl.sum(ones, 0)
        # Rebuild the map from the hot map, then point staged experts at
        # their spare rows. Same program, so the stores are ordered.
        for start in range(0, num_experts, MAP_BLOCK):
            offs = start + tl.arange(0, MAP_BLOCK)
            in_range = offs < num_experts
            hot = tl.load(hot_map_ptr + offs, mask=in_range, other=-1)
            tl.store(map_ptr + offs, hot, mask=in_range)
        tl.debug_barrier()
        tl.store(map_ptr + safe, (hot_slots + slot).to(tl.int32), mask=distinct)
        tl.store(gather_ptr + slot, cold, mask=distinct)
        tl.store(count_ptr, count.to(tl.int32))

    _KERNELS["plan"] = staging_plan
    return staging_plan


def _copy_kernel():
    """One launch for six bank tensors; unused slots exit immediately."""
    if "copy" in _KERNELS:
        return _KERNELS["copy"]
    from vllm.triton_utils import tl, triton

    @triton.jit
    def staging_copy(
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
        index_ptr,
        count_ptr,
        bytes0,
        bytes1,
        bytes2,
        bytes3,
        bytes4,
        bytes5,
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
        BLOCK: tl.constexpr,
    ):
        slot = tl.program_id(0)
        chunk = tl.program_id(1)
        which = tl.program_id(2)
        count = tl.load(count_ptr)
        if slot >= count:
            return
        src_row = tl.load(index_ptr + slot)
        offsets = chunk * BLOCK + tl.arange(0, BLOCK)
        if which == 0:
            mask = offsets < bytes0
            values = tl.load(src0 + src_row * sstride0 + offsets, mask=mask)
            tl.store(dst0 + slot * dstride0 + offsets, values, mask=mask)
        elif which == 1:
            mask = offsets < bytes1
            values = tl.load(src1 + src_row * sstride1 + offsets, mask=mask)
            tl.store(dst1 + slot * dstride1 + offsets, values, mask=mask)
        elif which == 2:
            mask = offsets < bytes2
            values = tl.load(src2 + src_row * sstride2 + offsets, mask=mask)
            tl.store(dst2 + slot * dstride2 + offsets, values, mask=mask)
        elif which == 3:
            mask = offsets < bytes3
            values = tl.load(src3 + src_row * sstride3 + offsets, mask=mask)
            tl.store(dst3 + slot * dstride3 + offsets, values, mask=mask)
        elif which == 4:
            mask = offsets < bytes4
            values = tl.load(src4 + src_row * sstride4 + offsets, mask=mask)
            tl.store(dst4 + slot * dstride4 + offsets, values, mask=mask)
        else:
            mask = offsets < bytes5
            values = tl.load(src5 + src_row * sstride5 + offsets, mask=mask)
            tl.store(dst5 + slot * dstride5 + offsets, values, mask=mask)

    _KERNELS["copy"] = staging_copy
    return staging_copy
