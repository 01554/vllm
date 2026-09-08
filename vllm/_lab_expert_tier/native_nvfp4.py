# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native NVFP4 bank adapter for the FreeToken wide-load Triton GEMV.

Orchestration follows FreeToken ``moe/fused_nvfp4.py`` at
af71ba43206e124f5ff6419b47ee36c6e9981078 (Apache-2.0).
Weights are checkpoint uint8, never Marlin-repacked int32. Multi-token inputs
use the same weight-only GEMV as a correctness fallback for prefill.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

Bank = Mapping[str, torch.Tensor]
_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def validate_bank(bank: Bank) -> tuple[int, int, int]:
    """Validate metadata only; return physical rows, hidden and intermediate."""
    rows = hidden = intermediate = 0
    device = None
    for prefix in ("w13", "w2"):
        packed = bank[f"{prefix}_weight"]
        scale = bank[f"{prefix}_weight_scale"]
        glob = bank[f"{prefix}_weight_scale_2"]
        if packed.dtype != torch.uint8 or packed.ndim != 3:
            raise TypeError("native NVFP4 requires raw uint8 [rows,N,K/2]")
        r, n, half_k = packed.shape
        k = 2 * half_k
        if r <= 0 or n <= 0 or k <= 0 or k % 16:
            raise ValueError("native NVFP4 needs positive sizes and K divisible by 16")
        if scale.dtype != torch.float8_e4m3fn:
            raise TypeError("block scales must be float8_e4m3fn")
        if scale.shape != (r, n, k // 16):
            raise ValueError("block scale shape must be [rows,N,K/16]")
        if glob.dtype not in (torch.float16, torch.float32):
            raise TypeError("per-row globals must be float16 or float32")
        if glob.shape != (r, n):
            raise ValueError("globals must preserve every output row: [rows,N]")
        for tensor in (packed, scale, glob):
            if not tensor.is_contiguous():
                raise ValueError("native bank tensors must be contiguous")
            if device is None:
                device = tensor.device
            if tensor.device != device:
                raise ValueError("all native bank tensors must share a device")
        if prefix == "w13":
            if n % 2:
                raise ValueError("gate/up bank must have equal consecutive halves")
            rows, hidden, intermediate = r, k, n // 2
        elif (r, n, k) != (rows, hidden, intermediate):
            raise ValueError("gate/up and down bank dimensions disagree")
    return rows, hidden, intermediate


@dataclass
class Workspace:
    num_experts: int
    num_rows: int
    hidden: int
    intermediate: int
    max_tokens: int
    top_k: int
    routes: torch.Tensor
    gate_up: torch.Tensor
    activated: torch.Tensor
    down: torch.Tensor
    output: torch.Tensor
    error: torch.Tensor


def allocate_workspace(
    bank: Bank, max_tokens: int, top_k: int, *, num_experts: int = 512
) -> Workspace:
    """Allocate persistent buffers and LUT before graph capture."""
    rows, hidden, intermediate = validate_bank(bank)
    if min(max_tokens, top_k, num_experts) <= 0:
        raise ValueError("workspace capacities must be positive")
    device = bank["w13_weight"].device
    workspace = Workspace(
        num_experts,
        rows,
        hidden,
        intermediate,
        max_tokens,
        top_k,
        torch.empty((max_tokens, top_k), dtype=torch.int32, device=device),
        torch.empty(
            (max_tokens, top_k, 2 * intermediate), dtype=torch.bfloat16, device=device
        ),
        torch.empty(
            (max_tokens * top_k, intermediate), dtype=torch.bfloat16, device=device
        ),
        torch.empty((max_tokens, top_k, hidden), dtype=torch.bfloat16, device=device),
        torch.empty((max_tokens, hidden), dtype=torch.bfloat16, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    )
    if device.type == "cuda":
        from . import native_nvfp4_kernels as kernels

        kernels.warmup(device)
    return workspace


def _cpu_projection(a, bank, prefix, routes, weights, *, routed_input=False):
    packed = bank[f"{prefix}_weight"].to(torch.int64)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    lut = torch.tensor(_LUT, dtype=torch.float32)
    matrix = lut[codes] * bank[f"{prefix}_weight_scale"].float().repeat_interleave(
        16, -1
    )
    glob = bank[f"{prefix}_weight_scale_2"].float()
    m, top_k = routes.shape
    out = torch.zeros((m, top_k, matrix.shape[1]), dtype=torch.bfloat16)
    for token in range(m):
        for route in range(top_k):
            row = int(routes[token, route])
            if row < 0:
                continue
            inp = a[token * top_k + route] if routed_input else a[token]
            result = (matrix[row] @ inp.float()) * glob[row]
            if weights is not None:
                result *= weights[token, route]
            out[token, route] = result.bfloat16()
    return out


def gemv(
    x: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    bank: Bank,
    step_map: torch.Tensor,
    workspace: Workspace,
    *,
    activation: str = "silu",
    routes_ready: bool = False,
) -> torch.Tensor:
    """Compute all routes, applying router weights once after down.

    ``ids=-1`` is padding. Missing/out-of-range active routes set the persistent
    ``workspace.error`` and contribute zero until the coordinator checks it.
    With ``routes_ready=True``, ``workspace.routes`` already contains physical
    row IDs; ``ids`` and ``step_map`` are retained for shape/device validation.
    Output aliases workspace storage and must be consumed before its next use.
    """
    if activation != "silu":
        raise NotImplementedError("native adapter currently supports SiLU only")
    m = x.shape[0]
    if x.ndim != 2 or x.shape[1] != workspace.hidden or x.dtype != torch.bfloat16:
        raise ValueError("x must be BF16 [tokens,hidden]")
    if m <= 0 or m > workspace.max_tokens:
        raise ValueError("token count exceeds native workspace capacity")
    if ids.shape != (m, workspace.top_k) or weights.shape != ids.shape:
        raise ValueError("route dimensions must match tokens and configured top_k")
    if ids.dtype != torch.int32 or weights.dtype != torch.float32:
        raise TypeError("ids must be int32 and router weights float32")
    if not routes_ready and (
        step_map.ndim != 1
        or step_map.numel() < workspace.num_experts
        or step_map.dtype != torch.int32
    ):
        raise ValueError("step_map must be int32 [num_experts] (optional sentinel)")
    for tensor in (x, weights, ids, step_map):
        if tensor.device != workspace.output.device or not tensor.is_contiguous():
            raise ValueError("inputs must be contiguous and on the workspace device")
    if validate_bank(bank) != (
        workspace.num_rows,
        workspace.hidden,
        workspace.intermediate,
    ):
        raise ValueError("bank dimensions changed after workspace allocation")
    if bank["w13_weight"].device != x.device:
        raise ValueError("bank device differs from workspace")
    routes = workspace.routes[:m]
    gu = workspace.gate_up[:m]
    act = workspace.activated[: m * workspace.top_k]
    down = workspace.down[:m]
    out = workspace.output[:m]
    if x.device.type == "cuda":
        from . import native_nvfp4_kernels as kernels

        route_ids = routes if routes_ready else ids
        route_map = None if routes_ready else step_map
        route_output = None
        if not routes_ready and not any(
            kernels._buffers_overlap(routes, tensor)
            for tensor in (x, weights, ids, step_map)
        ):
            route_output = routes
        kernels.launch_decode_gemm(
            x,
            bank["w13_weight"],
            bank["w13_weight_scale"],
            bank["w13_weight_scale_2"],
            gu,
            weights,
            route_ids,
            mul_routed_weight=False,
            a_row_is_route=False,
            num_rows=workspace.num_rows,
            expert_to_row=route_map,
            num_experts=workspace.num_experts if route_map is not None else None,
            error=workspace.error,
            route_output=route_output,
        )
        kernels.activation_inplace(gu, act)
        down_route_ids = routes if route_output is not None else route_ids
        down_route_map = None if route_output is not None else route_map
        kernels.launch_decode_gemm(
            act,
            bank["w2_weight"],
            bank["w2_weight_scale"],
            bank["w2_weight_scale_2"],
            down,
            weights,
            down_route_ids,
            mul_routed_weight=True,
            a_row_is_route=True,
            num_rows=workspace.num_rows,
            expert_to_row=down_route_map,
            num_experts=(workspace.num_experts if down_route_map is not None else None),
            error=workspace.error if down_route_map is not None else None,
            write_error=False,
        )
        kernels.sum_routes_inplace(down, out)
    else:
        if routes_ready:
            valid_row = (routes >= 0) & (routes < workspace.num_rows)
            bad = (routes != -1) & ~valid_row
        else:
            valid = (ids >= 0) & (ids < workspace.num_experts)
            mapped = step_map[torch.where(valid, ids, 0).long()]
            valid_row = valid & (mapped >= 0) & (mapped < workspace.num_rows)
            bad = (ids != -1) & ~valid_row
        workspace.error.bitwise_or_(bad.any().to(torch.int32))
        if routes_ready:
            routes.copy_(torch.where(valid_row, routes, -1))
        else:
            routes.copy_(torch.where(valid_row, mapped, -1))
        gu.copy_(_cpu_projection(x, bank, "w13", routes, None))
        gate, up = gu.float().chunk(2, dim=-1)
        act.copy_((torch.nn.functional.silu(gate) * up).reshape_as(act))
        down.copy_(_cpu_projection(act, bank, "w2", routes, weights, routed_input=True))
        total = torch.zeros((m, workspace.hidden), dtype=torch.float32)
        for route in range(workspace.top_k):
            total.add_(down[:, route].float())
        out.copy_(total)
    return out
