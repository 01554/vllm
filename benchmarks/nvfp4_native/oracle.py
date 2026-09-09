# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent CPU oracle for the NVFP4 MoE microbenchmark.

Plain torch only; it shares no code with the native loader or kernels.
Rounding sequence (agreed 2026-09-09): E2M1 nibble decode (low nibble
first), E4M3 block scale per 16 elements, FP32 projection with the global
scale, BF16 round of gate/up, FP32 SiLU and multiply, BF16 round of the
activation, FP32 down projection with its global scale and one application
of the router weight, BF16 round per route, FP32 sum over routes, final
BF16 round.
"""

from __future__ import annotations

import torch

# E2M1 code -> value (sign in bit 3).
_E2M1 = torch.tensor(
    [
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
    ],
    dtype=torch.float32,
)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] -> float32 [..., K]; low nibble is the even element."""
    if packed.dtype != torch.uint8:
        raise TypeError("packed NVFP4 weights must be uint8")
    low = (packed & 0x0F).long()
    high = (packed >> 4).long()
    values = torch.stack((_E2M1[low], _E2M1[high]), dim=-1)
    return values.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def dequantize_blocks(packed: torch.Tensor, block_scale: torch.Tensor) -> torch.Tensor:
    """One expert's weight [N, K] in float32 with block scales applied only.

    packed: uint8 [N, K/2]; block_scale: float8_e4m3fn [N, K/16]. The global
    scale is NOT folded in here: the rounding-aware path applies it after
    the FP32 projection (see expert_forward), which is where the kernels
    apply it.
    """
    values = unpack_e2m1(packed)
    n, k = values.shape
    if tuple(block_scale.shape) != (n, k // 16):
        raise ValueError(
            f"block scale shape {tuple(block_scale.shape)} != {(n, k // 16)}"
        )
    scales = block_scale.to(torch.float32).repeat_interleave(16, dim=-1)
    return values * scales


def dequantize_rows(
    packed: torch.Tensor, block_scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Source-semantics reference: fully dequantized float32 weight [N, K].

    Not used by the rounding-aware oracle path (global folded into the
    weight changes FP32 accumulation rounding); kept for weight inspection.
    """
    g = global_scale.to(torch.float32)
    if g.dim() == 1:
        g = g[:, None]
    return dequantize_blocks(packed, block_scale) * g


def expert_forward(
    x: torch.Tensor,
    w13_blocks: torch.Tensor,
    g13: torch.Tensor,
    w2_blocks: torch.Tensor,
    g2: torch.Tensor,
    route_weight: float,
) -> torch.Tensor:
    """One route for one token: x [K] bf16 -> [K] bf16 (router-weighted).

    Order: FP32 projection over block-scaled weights, then the global scale
    (per output row: g13 [2N], g2 [K]), BF16 round of gate/up, FP32 SiLU and
    multiply, BF16 round, FP32 down projection, global, router weight, BF16.
    """
    intermediate = w13_blocks.shape[0] // 2
    gate_up = (x.to(torch.float32) @ w13_blocks.t()) * g13.to(torch.float32)
    gate_up = gate_up.to(torch.bfloat16).to(torch.float32)
    gate, up = gate_up[:intermediate], gate_up[intermediate:]
    act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).to(torch.float32)
    down = (act @ w2_blocks.t()) * g2.to(torch.float32) * route_weight
    return down.to(torch.bfloat16)


def _row_globals(global_scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-output-row global scale [rows] from a scalar, a (gate, up) pair or [rows]."""
    g = global_scale.to(torch.float32).reshape(-1)
    if g.numel() == 1:
        return g.repeat(rows)
    if g.numel() == 2 and rows % 2 == 0:
        half = rows // 2
        return torch.cat((g[0].repeat(half), g[1].repeat(half)))
    if g.numel() == rows:
        return g
    raise ValueError(f"cannot map global scale of size {g.numel()} onto {rows} rows")


def moe_forward(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
) -> torch.Tensor:
    """Dense reference for M tokens; ids of -1 are padding routes.

    Experts are processed one at a time (bounded memory: one dequantized
    expert at a time plus [M, top_k, K] BF16 route outputs); the per-token
    sum over routes is taken in route order in FP32 and rounded to BF16.
    """
    m, top_k = topk_ids.shape
    k = x.shape[1]
    route_out = torch.zeros((m, top_k, k), dtype=torch.bfloat16)
    ids = topk_ids.to(torch.int64)
    for e in torch.unique(ids[ids >= 0]).tolist():
        w13_blocks = dequantize_blocks(w13_packed[e], w13_scale[e])
        g13 = _row_globals(w13_scale_2[e], w13_blocks.shape[0])
        w2_blocks = dequantize_blocks(w2_packed[e], w2_scale[e])
        g2 = _row_globals(w2_scale_2[e], w2_blocks.shape[0])
        for t, r in torch.nonzero(ids == e).tolist():
            route_out[t, r] = expert_forward(
                x[t], w13_blocks, g13, w2_blocks, g2, float(topk_weights[t, r])
            )
    out = torch.zeros((m, k), dtype=torch.float32)
    for r in range(top_k):
        out += route_out[:, r].to(torch.float32)
    return out.to(torch.bfloat16)
