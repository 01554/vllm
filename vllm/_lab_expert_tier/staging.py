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
sequence can be captured in a CUDA Graph and replayed with different routing:

- `plan_staging` derives, from this step's routing IDs and the current cold
  and hot maps, the distinct selected cold experts (invalid and padding IDs
  ignored, duplicates collapsed), a gather index per staging slot, the
  device count of staged experts, and the step's expert map. The map is
  rebuilt from the hot map every step, so a previous step's staging
  assignment can never leak into the next.
- `gather_staging` copies exactly `count` rows of every bank tensor from the
  cold source to the staging rows. On CUDA a Triton kernel reads the count
  on the device and exits early for unused slots, which is what keeps a
  zero-miss step from paying for `top_k` copies. The reference path is
  Python and is used for CPU tests and non-CUDA devices.

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


def plan_staging(ids, cold_map, hot_map, hot_slots, staging_slots):
    """Fixed-shape staging plan for one forward.

    Args:
        ids: [rows, k] routing IDs; -1 marks padding. rows * k must not
            exceed staging_slots.
        cold_map: [E] int32 device map expert -> cold RAM slot or -1.
        hot_map: [E] int32 device map expert -> hot VRAM slot or -1.
        hot_slots: number of resident hot rows; staging rows follow them.
        staging_slots: spare rows at the end of the hot bank.

    Returns:
        gather_index: [staging_slots] int64 cold slots to copy, in slot
            order; entries at or past `count` are unused and hold 0.
        expert_map: [E] int32 this step's map: hot experts keep their hot
            slot, staged cold experts map to hot_slots + staging slot, and
            every other expert maps to -1.
        count: [] int32 device scalar, the number of staged experts.
    """
    import torch

    num_experts = hot_map.shape[0]
    flat = ids.reshape(-1)
    if flat.shape[0] > staging_slots:
        raise ValueError("Forward selects more slots than the staging rows")
    valid = (flat >= 0) & (flat < num_experts)
    # Clamp before any gather so an invalid ID can never index out of range;
    # its result is masked out below.
    safe = torch.where(valid, flat, torch.zeros_like(flat)).to(torch.int64)
    cold_slot = cold_map[safe].to(torch.int64)
    is_cold = valid & (cold_slot >= 0)
    # Sort the cold expert IDs (non-cold entries sort last as `num_experts`)
    # so duplicates become adjacent; the first of each run is distinct.
    key = torch.where(is_cold, safe, torch.full_like(safe, num_experts))
    sorted_key, order = torch.sort(key)
    sorted_cold_slot = cold_slot[order]
    first = torch.ones_like(sorted_key, dtype=torch.bool)
    first[1:] = sorted_key[1:] != sorted_key[:-1]
    distinct = first & (sorted_key < num_experts)
    slot = torch.cumsum(distinct.to(torch.int64), 0) - 1
    count = distinct.sum().to(torch.int32)
    # Scatter through one spare "dummy" index so non-distinct positions land
    # nowhere visible; that keeps every shape static.
    gather = torch.zeros(staging_slots + 1, dtype=torch.int64, device=ids.device)
    gather.scatter_(
        0,
        torch.where(distinct, slot, torch.full_like(slot, staging_slots)),
        sorted_cold_slot,
    )
    # Device fills only: a Python scalar store is a host-to-device copy,
    # which CUDA Graph capture rejects.
    expert_map = torch.full(
        (num_experts + 1,), -1, dtype=torch.int32, device=ids.device
    )
    expert_map[:num_experts].copy_(hot_map)
    expert_map.scatter_(
        0,
        torch.where(distinct, sorted_key, torch.full_like(sorted_key, num_experts)),
        (hot_slots + slot).to(torch.int32),
    )
    return gather[:staging_slots], expert_map[:num_experts], count


def gather_staging_reference(source, staging, gather_index, count):
    """Copy `count` rows of every tensor; host-synchronizing, for tests."""
    rows = int(count.item())
    for name in TENSORS:
        for slot in range(rows):
            staging[name][slot].copy_(source[name][gather_index[slot]])


def gather_staging(source, staging, gather_index, count):
    """Copy the planned cold rows into the staging rows of every bank tensor.

    `source[name]` is the cold bank as a device-visible tensor (UVA view of
    pinned RAM) and `staging[name]` the [staging_slots, ...] view at the end
    of the hot bank. On CUDA the copy reads `count` on the device and never
    synchronizes; elsewhere the reference path is used.
    """
    if source[TENSORS[0]].device.type != "cuda":
        gather_staging_reference(source, staging, gather_index, count)
        return
    kernel = _staged_copy_kernel()
    for name in TENSORS:
        src = _byte_rows(source[name])
        dst = _byte_rows(staging[name])
        if src.shape[1] != dst.shape[1]:
            raise ValueError(f"{name}: staging row size differs from the source")
        row_bytes = dst.shape[1]
        block = 4096
        grid = (dst.shape[0], (row_bytes + block - 1) // block)
        kernel[grid](
            src,
            dst,
            gather_index,
            count,
            row_bytes,
            src.stride(0),
            dst.stride(0),
            BLOCK=block,
        )


def _byte_rows(tensor):
    """View a [rows, ...] contiguous tensor as [rows, bytes] uint8."""
    import torch

    if not tensor.is_contiguous():
        raise ValueError("Staging requires contiguous bank rows")
    rows = tensor.shape[0]
    return tensor.view(torch.uint8).reshape(rows, -1)


_KERNEL: Any = None


def _staged_copy_kernel():
    """Build the Triton row-copy kernel once; unused slots exit immediately."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    from vllm.triton_utils import tl, triton

    @triton.jit
    def staged_copy(
        src_ptr,
        dst_ptr,
        index_ptr,
        count_ptr,
        row_bytes,
        src_stride,
        dst_stride,
        BLOCK: tl.constexpr,
    ):
        slot = tl.program_id(0)
        chunk = tl.program_id(1)
        count = tl.load(count_ptr)
        if slot >= count:
            return
        src_row = tl.load(index_ptr + slot)
        offsets = chunk * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < row_bytes
        values = tl.load(src_ptr + src_row * src_stride + offsets, mask=mask)
        tl.store(dst_ptr + slot * dst_stride + offsets, values, mask=mask)

    _KERNEL = staged_copy
    return _KERNEL
