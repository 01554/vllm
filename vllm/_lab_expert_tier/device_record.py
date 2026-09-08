# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch routing record for the device observer.

The runtime's per-layer routing checks and `DeviceObserver.record_layer` /
`DeviceHeatAccumulator.record_layer` are a chain of about thirty small
elementwise, reduction, and index kernels per layer (bool masks, compares,
`any`, `index_add_`, three record copies, a device assert). FreeToken keeps
the same bookkeeping inside its LRU kernel. This module writes the same
tensors, with the same meaning, in one program per layer:

- the observer's ids / activity / valid records for this layer's rows;
- the accumulator's per-layer counts, step route totals, valid-token
  count, expected-layer counter, hot-map availability flags, and the
  sticky error flag (including the runtime's routing checks, which used
  to be a device assertion: -1 only on padding rows, real ids in range,
  weights finite and nonnegative);
- at layer 0, the per-step resets the accumulator performs.

`RecordTargets` names the tensors; the observer hands them out and does
its host bookkeeping in `note_kernel_record`. The torch reference here is
the contract the CPU tests hold against the observer's own path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_LANES = 1024


@dataclass(frozen=True)
class RecordTargets:
    """Device tensors of one observer, all at fixed addresses."""

    num_layers: int
    num_experts: int
    top_k: int
    ids_record: Any  # [capacity, L, K] int64
    activity_record: Any  # [capacity, L, K] bool
    valid_record: Any  # [capacity, L] bool
    counts: Any  # [L, E] int64
    step_route_total: Any  # [] int64
    step_route_hot: Any  # [] int64
    step_valid_tokens: Any  # [] int64
    expected_layer: Any  # [] int64
    error_flag: Any  # [] bool
    step_hot_map_seen: Any  # [L] bool
    step_hot_map_missing: Any  # [] bool


def record_reference(targets, layer, rows, ids, weights, padding, hot_map):
    """Torch reference of the fused record (host-synchronizing).

    Mirrors `DeviceHeatAccumulator.record_layer` plus `DeviceObserver`'s
    record copies plus the runtime's routing checks folded into the error
    flag. `weights != 0` is the activity flag; `~padding` the valid mask.
    """
    import torch

    t = targets
    E, L = t.num_experts, t.num_layers
    ids_view = ids[:rows].to(torch.int64)
    weights_view = weights[:rows]
    padding_view = padding[:rows]
    valid_view = ~padding_view
    active_view = weights_view != 0
    if layer == 0:
        t.error_flag.logical_or_(t.expected_layer.ne(0) & t.expected_layer.ne(L))
        t.counts.zero_()
        t.step_route_total.zero_()
        t.step_route_hot.zero_()
        t.step_valid_tokens.zero_()
        t.expected_layer.zero_()
        t.step_hot_map_seen.zero_()
        t.step_hot_map_missing.zero_()
    t.error_flag.logical_or_(t.expected_layer.ne(layer))
    if hot_map is None:
        t.step_hot_map_missing.fill_(True)
    else:
        t.step_hot_map_seen[layer].fill_(True)
    t.ids_record[:rows, layer].copy_(ids_view)
    t.activity_record[:rows, layer].copy_(active_view)
    t.valid_record[:rows, layer].copy_(valid_view)
    valid_count = valid_view.to(torch.int64).sum()
    if layer == 0:
        t.step_valid_tokens.copy_(valid_count)
    else:
        t.error_flag.logical_or_(valid_count.ne(t.step_valid_tokens))
    in_range = (ids_view >= 0) & (ids_view < E)
    invalid_real = (~in_range) & valid_view[:, None]
    t.error_flag.logical_or_(invalid_real.any())
    # The runtime's former device assertion.
    allowed = in_range | ((ids_view == -1) & padding_view[:, None])
    good_weights = (torch.isfinite(weights_view) & (weights_view >= 0)) | (
        padding_view[:, None]
    )
    t.error_flag.logical_or_(~(allowed & good_weights).all())
    safe_ids = ids_view.clamp(0, E - 1)
    selected = active_view & valid_view[:, None] & in_range
    t.counts[layer].view(-1).index_add_(
        0, safe_ids.reshape(-1), selected.reshape(-1).to(torch.int64)
    )
    t.step_route_total.add_(selected.to(torch.int64).sum())
    if hot_map is not None:
        t.step_route_hot.add_(
            (selected & hot_map[safe_ids].ge(0)).to(torch.int64).sum()
        )
    t.expected_layer.add_(1)


def record(targets, layer, rows, ids, weights, padding, hot_map):
    """Fused record: Triton on CUDA, the reference elsewhere."""
    if ids.device.type != "cuda":
        record_reference(targets, layer, rows, ids, weights, padding, hot_map)
        return
    import torch

    t = targets
    K = t.top_k
    lanes = rows * K
    if lanes > MAX_LANES:
        raise ValueError("Fused record supports at most MAX_LANES routes")
    if ids.shape[1] != K or weights.shape[1] != K:
        raise ValueError("Routing width differs from the observer's top_k")
    ids_c = ids.contiguous()
    weights_c = weights.contiguous()
    padding_c = padding.contiguous().view(torch.uint8)
    has_map = hot_map is not None
    map_c = hot_map.contiguous() if has_map else t.expected_layer
    _record_kernel()[(1,)](
        ids_c,
        weights_c,
        padding_c,
        map_c,
        t.ids_record,
        t.activity_record.view(torch.uint8),
        t.valid_record.view(torch.uint8),
        t.counts,
        t.step_route_total.reshape(1),
        t.step_route_hot.reshape(1),
        t.step_valid_tokens.reshape(1),
        t.expected_layer.reshape(1),
        t.error_flag.view(torch.uint8).reshape(1),
        t.step_hot_map_seen.view(torch.uint8),
        t.step_hot_map_missing.view(torch.uint8).reshape(1),
        layer,
        rows,
        t.num_layers,
        t.num_experts,
        t.ids_record.stride(0),
        t.ids_record.stride(1),
        t.valid_record.stride(0),
        K=K,
        LANES=_next_power_of_two(lanes),
        HAS_MAP=has_map,
        BLOCK=1024,
    )


def _next_power_of_two(value):
    return 1 << max(int(value) - 1, 0).bit_length()


_KERNELS: dict[str, Any] = {}


def _record_kernel():
    if "record" in _KERNELS:
        return _KERNELS["record"]
    from vllm.triton_utils import tl, triton

    @triton.jit
    def observer_record(
        ids_ptr,
        weights_ptr,
        padding_ptr,
        map_ptr,
        ids_record_ptr,
        activity_record_ptr,
        valid_record_ptr,
        counts_ptr,
        route_total_ptr,
        route_hot_ptr,
        valid_tokens_ptr,
        expected_ptr,
        error_ptr,
        map_seen_ptr,
        map_missing_ptr,
        layer,
        rows,
        num_layers,
        num_experts,
        rec_stride_row,
        rec_stride_layer,
        valid_stride_row,
        K: tl.constexpr,
        LANES: tl.constexpr,
        HAS_MAP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        layer64 = tl.full((), 0, tl.int64) + layer
        expected = tl.load(expected_ptr)
        error = tl.load(error_ptr) != 0
        if layer64 == 0:
            error = error | ((expected != 0) & (expected != num_layers))
            total = num_layers * num_experts
            for start in range(0, total, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                tl.store(
                    counts_ptr + offs,
                    tl.zeros((BLOCK,), dtype=tl.int64),
                    mask=offs < total,
                )
            tl.store(route_total_ptr, tl.full((), 0, tl.int64))
            tl.store(route_hot_ptr, tl.full((), 0, tl.int64))
            tl.store(valid_tokens_ptr, tl.full((), 0, tl.int64))
            expected = tl.full((), 0, tl.int64)
            for start in range(0, num_layers, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                tl.store(
                    map_seen_ptr + offs,
                    tl.zeros((BLOCK,), dtype=tl.int8),
                    mask=offs < num_layers,
                )
            tl.store(map_missing_ptr, tl.full((), 0, tl.int8))
        error = error | (expected != layer64)
        if HAS_MAP:
            tl.store(map_seen_ptr + layer64, tl.full((), 1, tl.int8))
        else:
            tl.store(map_missing_ptr, tl.full((), 1, tl.int8))
        tl.debug_barrier()
        lane = tl.arange(0, LANES)
        row = lane // K
        col = lane - row * K
        present = row < rows
        ids = tl.load(ids_ptr + lane, mask=present, other=-1).to(tl.int64)
        weights = tl.load(weights_ptr + lane, mask=present, other=0.0)
        padding = tl.load(padding_ptr + row, mask=present, other=1) != 0
        valid = present & (~padding)
        active = present & (weights != 0)
        in_range = present & (ids >= 0) & (ids < num_experts)
        # Records for this layer's rows.
        rec = row * rec_stride_row + layer64 * rec_stride_layer + col
        tl.store(ids_record_ptr + rec, ids, mask=present)
        tl.store(activity_record_ptr + rec, active.to(tl.int8), mask=present)
        first = present & (col == 0)
        tl.store(
            valid_record_ptr + row * valid_stride_row + layer64,
            valid.to(tl.int8),
            mask=first,
        )
        valid_count = tl.sum((valid & first).to(tl.int64), 0)
        if layer64 == 0:
            tl.store(valid_tokens_ptr, valid_count)
        else:
            error = error | (valid_count != tl.load(valid_tokens_ptr))
        invalid_real = (~in_range) & valid
        error = error | (tl.sum(invalid_real.to(tl.int32), 0) > 0)
        allowed = in_range | ((ids == -1) & padding)
        # finite (not NaN, not +/-inf) and nonnegative, or a padding row.
        finite = (weights == weights) & (tl.abs(weights) < float("inf"))
        good_weights = (finite & (weights >= 0)) | padding
        bad = present & (~(allowed & good_weights))
        error = error | (tl.sum(bad.to(tl.int32), 0) > 0)
        safe = tl.where(in_range, ids, 0)
        selected = active & valid & in_range
        tl.atomic_add(
            counts_ptr + layer64 * num_experts + safe,
            selected.to(tl.int64),
            mask=selected,
        )
        tl.store(
            route_total_ptr, tl.load(route_total_ptr) + tl.sum(selected.to(tl.int64), 0)
        )
        if HAS_MAP:
            resident = tl.load(map_ptr + safe, mask=selected, other=-1) >= 0
            hot = tl.sum((selected & resident).to(tl.int64), 0)
            tl.store(route_hot_ptr, tl.load(route_hot_ptr) + hot)
        tl.store(expected_ptr, expected + 1)
        tl.store(error_ptr, error.to(tl.int8))

    _KERNELS["record"] = observer_record
    return observer_record


__all__ = ["MAX_LANES", "RecordTargets", "record", "record_reference"]
