# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped native NVFP4 prefill adapter.

The grouped GEMM follows FreeToken ``moe/fused_nvfp4.py`` and
``kernel/triton/nvfp4_fused_moe.py`` at commit
``af71ba43206e124f5ff6419b47ee36c6e9981078`` (Apache-2.0).  It aligns logical
routes before mapping aligned blocks to physical bank rows, which permits
resident banks whose row count is larger than the logical expert count.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .native_nvfp4 import Bank, validate_bank
from .native_nvfp4 import Workspace as DecodeWorkspace


@dataclass
class Workspace(DecodeWorkspace):
    """Persistent storage for one grouped prefill configuration."""

    logical_routes: torch.Tensor
    sorted_token_ids: torch.Tensor
    logical_expert_ids: torch.Tensor
    physical_expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor


def allocate_workspace(
    bank: Bank, max_tokens: int, top_k: int, *, num_experts: int = 512
) -> Workspace:
    """Allocate all route, alignment, and intermediate buffers before forward."""

    rows, hidden, intermediate = validate_bank(bank)
    if not isinstance(max_tokens, int) or not isinstance(top_k, int):
        raise TypeError("workspace capacities must be integers")
    if not isinstance(num_experts, int):
        raise TypeError("num_experts must be an integer")
    if min(max_tokens, top_k, num_experts) <= 0:
        raise ValueError("workspace capacities must be positive")

    device = bank["w13_weight"].device
    route_capacity = max_tokens * top_k
    # FreeToken uses BLOCK_M=16 for short batches and 32 for larger batches.
    # Reserve for the largest alignment expansion and enough block IDs for the
    # smaller BLOCK_M configuration.
    max_sorted = route_capacity + num_experts * (32 - 1)
    # The alignment op writes complete blocks.  Reserve a complete final
    # BLOCK_M block for either grouped-kernel configuration.
    max_sorted = ((max_sorted + 32 - 1) // 32) * 32
    max_blocks = (max_sorted + 16 - 1) // 16
    workspace = Workspace(
        num_experts=num_experts,
        num_rows=rows,
        hidden=hidden,
        intermediate=intermediate,
        max_tokens=max_tokens,
        top_k=top_k,
        routes=torch.empty((max_tokens, top_k), dtype=torch.int32, device=device),
        gate_up=torch.empty(
            (max_tokens, top_k, 2 * intermediate),
            dtype=torch.bfloat16,
            device=device,
        ),
        activated=torch.empty(
            (route_capacity, intermediate),
            dtype=torch.bfloat16,
            device=device,
        ),
        down=torch.empty(
            (max_tokens, top_k, hidden), dtype=torch.bfloat16, device=device
        ),
        output=torch.empty((max_tokens, hidden), dtype=torch.bfloat16, device=device),
        error=torch.zeros(1, dtype=torch.int32, device=device),
        logical_routes=torch.empty(
            (max_tokens, top_k), dtype=torch.int32, device=device
        ),
        sorted_token_ids=torch.empty((max_sorted,), dtype=torch.int32, device=device),
        logical_expert_ids=torch.empty((max_blocks,), dtype=torch.int32, device=device),
        physical_expert_ids=torch.empty(
            (max_blocks,), dtype=torch.int32, device=device
        ),
        num_tokens_post_padded=torch.empty((1,), dtype=torch.int32, device=device),
    )
    if device.type == "cuda":
        from . import native_prefill_kernels as kernels

        kernels.warmup(device)
    return workspace


def _validate_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    step_map: torch.Tensor,
    workspace: Workspace,
) -> int:
    if x.ndim != 2 or x.shape[1] != workspace.hidden:
        raise ValueError("x must be [tokens, hidden]")
    if x.dtype != torch.bfloat16:
        raise TypeError("native prefill activations must be BF16")
    tokens = x.shape[0]
    if tokens <= 0 or tokens > workspace.max_tokens:
        raise ValueError("token count exceeds prefill workspace capacity")
    if ids.shape != (tokens, workspace.top_k):
        raise ValueError("route IDs must be [tokens, top_k]")
    if weights.shape != ids.shape:
        raise ValueError("router weights must match route IDs")
    if ids.dtype != torch.int32:
        raise TypeError("route IDs must be int32")
    if weights.dtype != torch.float32:
        raise TypeError("router weights must be float32")
    if step_map.ndim != 1 or step_map.numel() < workspace.num_experts:
        raise ValueError("step_map must contain all logical experts")
    if step_map.dtype != torch.int32:
        raise TypeError("step_map must be int32")
    device = workspace.output.device
    for tensor in (x, weights, ids, step_map):
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError("prefill inputs must be contiguous on one device")
    return tokens


def _alignment_capacity(routes: int, num_experts: int, block_size: int) -> int:
    capacity = routes + num_experts * (block_size - 1)
    if routes < num_experts:
        capacity = min(routes * block_size, capacity)
    return ((capacity + block_size - 1) // block_size) * block_size


def _align_routes(
    logical_routes: torch.Tensor,
    workspace: Workspace,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run vLLM's alignment op into persistent workspace buffers."""

    from vllm import _custom_ops as ops

    capacity = _alignment_capacity(
        logical_routes.numel(), workspace.num_experts, block_size
    )
    num_blocks = (capacity + block_size - 1) // block_size
    sorted_ids = workspace.sorted_token_ids[:capacity]
    logical_expert_ids = workspace.logical_expert_ids[:num_blocks]
    # The logical routes have already been range-checked and sanitized.  A map
    # argument here would make the histogram index physical rows, which can be
    # larger than num_experts and is unsafe for pooled banks.
    ops.moe_align_block_size(
        logical_routes,
        workspace.num_experts,
        block_size,
        sorted_ids,
        logical_expert_ids,
        workspace.num_tokens_post_padded,
        None,
    )
    return sorted_ids, logical_expert_ids


def _prefill_cuda(
    x: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    bank: Bank,
    step_map: torch.Tensor,
    workspace: Workspace,
    tokens: int,
) -> torch.Tensor:
    from . import native_nvfp4_kernels as decode_kernels
    from . import native_prefill_kernels as kernels

    logical = workspace.logical_routes[:tokens]
    routes = workspace.routes[:tokens]
    decode_kernels.map_routes_inplace(
        ids,
        None,
        logical,
        num_experts=workspace.num_experts,
        num_rows=workspace.num_experts,
        error=workspace.error,
    )
    decode_kernels.map_routes_inplace(
        logical,
        step_map,
        routes,
        num_experts=workspace.num_experts,
        num_rows=workspace.num_rows,
        error=workspace.error,
    )

    config = kernels.prefill_config(tokens)
    sorted_ids, logical_expert_ids = _align_routes(
        logical, workspace, block_size=config["BLOCK_SIZE_M"]
    )
    num_blocks = logical_expert_ids.numel()
    physical_expert_ids = workspace.physical_expert_ids[:num_blocks]
    kernels.map_aligned_experts_inplace(
        logical_expert_ids,
        physical_expert_ids,
        step_map,
        workspace.num_tokens_post_padded,
        block_size=config["BLOCK_SIZE_M"],
        num_experts=workspace.num_experts,
        num_rows=workspace.num_rows,
        error=workspace.error,
    )

    route_count = tokens * workspace.top_k
    # Alignment deliberately omits padding routes.  Clear every output scratch
    # before each launch so omitted blocks cannot expose a previous call.
    gate_up = workspace.gate_up[:tokens]
    activated = workspace.activated[:route_count]
    down = workspace.down[:tokens]
    gate_up.zero_()
    activated.zero_()
    down.zero_()
    workspace.output[:tokens].zero_()

    weights_flat = weights.reshape(-1)
    kernels.launch_prefill_gemm(
        x,
        bank["w13_weight"],
        bank["w13_weight_scale"],
        bank["w13_weight_scale_2"],
        gate_up.reshape(route_count, 2 * workspace.intermediate),
        weights_flat,
        sorted_ids,
        physical_expert_ids,
        workspace.num_tokens_post_padded,
        num_valid_tokens=route_count,
        kernel_top_k=workspace.top_k,
        mul_routed_weight=False,
        config=config,
        num_rows=workspace.num_rows,
    )
    decode_kernels.activation_inplace(gate_up, activated)
    kernels.launch_prefill_gemm(
        activated,
        bank["w2_weight"],
        bank["w2_weight_scale"],
        bank["w2_weight_scale_2"],
        down.reshape(route_count, workspace.hidden),
        weights_flat,
        sorted_ids,
        physical_expert_ids,
        workspace.num_tokens_post_padded,
        num_valid_tokens=route_count,
        kernel_top_k=1,
        mul_routed_weight=True,
        config=config,
        num_rows=workspace.num_rows,
    )
    decode_kernels.sum_routes_inplace(down, workspace.output[:tokens])
    return workspace.output[:tokens]


def prefill(
    x: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    bank: Bank,
    step_map: torch.Tensor,
    workspace: Workspace,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    """Run grouped inline-dequant NVFP4 prefill.

    ``ids == -1`` is padding.  Other invalid logical IDs and missing physical
    rows set ``workspace.error`` and contribute zero.  The router weight is
    applied once by the down projection before route reduction.
    """

    if activation != "silu":
        raise NotImplementedError("native prefill currently supports SiLU only")
    tokens = _validate_inputs(x, weights, ids, step_map, workspace)
    if validate_bank(bank) != (
        workspace.num_rows,
        workspace.hidden,
        workspace.intermediate,
    ):
        raise ValueError("bank dimensions changed after workspace allocation")
    if bank["w13_weight"].device != x.device:
        raise ValueError("bank and activations must be on the same device")
    if x.device.type == "cuda":
        return _prefill_cuda(x, weights, ids, bank, step_map, workspace, tokens)

    # The serial CPU reference uses the exact same LUT, global-scale placement,
    # BF16 casts, and route-order reduction as the native GEMV adapter.
    from .native_nvfp4 import gemv

    return gemv(x, weights, ids, bank, step_map, workspace, activation=activation)


__all__ = ["Workspace", "allocate_workspace", "prefill"]
