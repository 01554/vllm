# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the pool layer helpers: route masking, block-to-row remap,
and the fixed-grid copy reference."""

import torch

from vllm.model_executor.layers.fused_moe.expert_pool.copy import copy_rows
from vllm.model_executor.layers.fused_moe.expert_pool.layer import (
    marlin_block_size,
    mask_routes,
    physical_block_experts,
    physical_block_experts_device,
)
from vllm.model_executor.layers.fused_moe.expert_pool.tables import TENSORS


def test_mask_routes_hides_absent_experts_and_keeps_padding():
    expert_map = torch.tensor([5, -1, 7, -1], dtype=torch.int32)  # rows for 0, 2
    ids = torch.tensor([[0, 1, 2, 3], [-1, 2, 9, 0]], dtype=torch.int32)
    out = mask_routes(ids, expert_map)
    assert out.tolist() == [[0, -1, 2, -1], [-1, 2, -1, 0]]


def test_physical_block_experts_maps_used_blocks_only():
    expert_map = torch.tensor([10, -1, 12], dtype=torch.int32)
    logical = torch.tensor([2, 0, 7, 2], dtype=torch.int32)  # block 2 is garbage
    post_padded = torch.tensor([16], dtype=torch.int32)  # 2 blocks of 8 used
    rows = physical_block_experts(logical, post_padded, 8, expert_map, 3)
    assert rows.tolist() == [12, 10, -1, -1]


def test_marlin_block_size_matches_stock_choice_shape():
    assert marlin_block_size(1, 10, 9984, 512, None) == 8
    assert marlin_block_size(4096, 10, 512, 512, None) == 64
    assert marlin_block_size(1, 10, 512, 512, torch.int8) >= 16


def test_copy_rows_reference_copies_every_tensor_in_order():
    src = {n: torch.arange(8 * 4, dtype=torch.int32).reshape(8, 4) for n in TENSORS}
    dst = {n: torch.zeros(3, 4, dtype=torch.int32) for n in TENSORS}
    src_rows = torch.tensor([6, 1, 0, 0], dtype=torch.int32)
    dst_rows = torch.tensor([2, 0, 0, 0], dtype=torch.int32)
    copy_rows(src, dst, src_rows, dst_rows, torch.tensor([2], dtype=torch.int32))
    for n in TENSORS:
        assert torch.equal(dst[n][2], src[n][6]) and torch.equal(dst[n][0], src[n][1])
        assert int(dst[n][1].sum()) == 0


def test_physical_block_experts_device_falls_back_to_torch_on_cpu():
    expert_map = torch.tensor([10, -1, 12], dtype=torch.int32)
    logical = torch.tensor([2, 0, 7, 2], dtype=torch.int32)
    post_padded = torch.tensor([16], dtype=torch.int32)
    a = physical_block_experts(logical, post_padded, 8, expert_map, 3)
    b = physical_block_experts_device(logical, post_padded, 8, expert_map, 3)
    assert torch.equal(a, b)


def test_physical_block_experts_kernel_matches_torch_on_cuda():
    if not torch.cuda.is_available():
        return
    import random

    device = torch.device("cuda")
    rng = random.Random(11)
    for _ in range(20):
        E = rng.choice([8, 64, 512])
        bank_rows = E + rng.randint(1, 3 * E)  # bank larger than the expert count
        n_blocks = rng.randint(1, 300)
        block = rng.choice([8, 16, 32, 64])
        # Garbage ids (out of range, negative) beyond post_padded and inside.
        logical = torch.randint(-5, E + 5, (n_blocks,), dtype=torch.int32)
        post_padded = torch.tensor(
            [rng.randint(0, n_blocks * block)], dtype=torch.int32
        )
        expert_map = torch.randint(0, bank_rows, (E,), dtype=torch.int32)
        expert_map[torch.rand(E) < 0.3] = -1  # absent experts
        ref = physical_block_experts(logical, post_padded, block, expert_map, E)
        got = physical_block_experts_device(
            logical.to(device), post_padded.to(device), block, expert_map.to(device), E
        ).cpu()
        assert torch.equal(ref, got)
