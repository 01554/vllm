# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical route grouping and physical block mapping for small decode batches."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _small_align(
    ids,
    mapping,
    sorted_ids,
    block_ids,
    post_pad,
    L: tl.constexpr,
    E: tl.constexpr,
    ROWS: tl.constexpr,
    B: tl.constexpr,
    W: tl.constexpr,
    OUT: tl.constexpr,
):
    lane = tl.arange(0, W)
    expert = tl.load(ids + lane, lane < L, other=-1)
    valid = (lane < L) & (expert >= 0) & (expert < E)
    physical = tl.load(mapping + expert, valid, other=-1)
    valid = valid & (physical >= 0) & (physical < ROWS)
    same = (expert[:, None] == expert[None, :]) & valid[None, :]
    count = tl.sum(same.to(tl.int32), 1)
    rank = tl.sum((same & (lane[None, :] < lane[:, None])).to(tl.int32), 1)
    leader = valid & (rank == 0)
    padded = tl.where(leader, tl.cdiv(count, B) * B, 0)
    start = tl.sum(tl.where(expert[None, :] < expert[:, None], padded[None, :], 0), 1)
    total = tl.sum(padded, 0)
    out = tl.arange(0, OUT)
    tl.store(sorted_ids + out, L, out < L * B)
    tl.store(block_ids + lane, -1, lane < L)
    tl.store(post_pad, total)
    # One CTA initializes the padding before scattering distinct route positions.
    tl.debug_barrier()
    tl.store(sorted_ids + start + rank, lane, valid)
    for chunk in range(triton.cdiv(L, B)):
        tl.store(
            block_ids + start // B + chunk,
            physical,
            leader & (chunk * B < count),
        )


def small_align(
    ids: torch.Tensor, expert_map: torch.Tensor, block_size: int, bank_rows: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return padded route IDs, physical block IDs, and padded length.

    Same-expert routes retain lane order. Invalid or absent routes are padding.
    Only contiguous CUDA inputs with 1..64 lanes are supported.
    """
    lanes = ids.numel()
    if not 0 < lanes <= 64 or block_size not in (8, 16, 32, 48, 64):
        raise ValueError("small alignment requires 1..64 lanes and a Marlin block")
    if not ids.is_contiguous() or not expert_map.is_contiguous():
        raise ValueError("small alignment requires contiguous inputs")
    if not ids.is_cuda or expert_map.device != ids.device:
        raise ValueError("small alignment requires inputs on the same CUDA device")
    sorted_ids = torch.empty(lanes * block_size, dtype=torch.int32, device=ids.device)
    blocks = torch.empty(lanes, dtype=torch.int32, device=ids.device)
    post_pad = torch.empty(1, dtype=torch.int32, device=ids.device)
    _small_align[(1,)](
        ids,
        expert_map,
        sorted_ids,
        blocks,
        post_pad,
        lanes,
        expert_map.numel(),
        bank_rows,
        block_size,
        triton.next_power_of_2(lanes),
        triton.next_power_of_2(lanes * block_size),
        num_warps=4,
    )
    return sorted_ids, blocks, post_pad
