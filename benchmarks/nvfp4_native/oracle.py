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


def dequantize_rows(
    packed: torch.Tensor, block_scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """One expert's weight [N, K] in float32 from the raw ModelOpt layout.

    packed: uint8 [N, K/2]; block_scale: float8_e4m3fn [N, K/16];
    global_scale: float32/float16 scalar or [N] (per output row).
    """
    values = unpack_e2m1(packed)
    n, k = values.shape
    if tuple(block_scale.shape) != (n, k // 16):
        raise ValueError(
            f"block scale shape {tuple(block_scale.shape)} != {(n, k // 16)}"
        )
    scales = block_scale.to(torch.float32).repeat_interleave(16, dim=-1)
    g = global_scale.to(torch.float32)
    if g.dim() == 1:
        g = g[:, None]
    return values * scales * g


def expert_forward(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    route_weight: float,
) -> torch.Tensor:
    """One route for one token: x [K] bf16 -> [K] bf16 (already router-weighted)."""
    intermediate = w13.shape[0] // 2
    gate_up = (x.to(torch.float32) @ w13.t()).to(torch.bfloat16).to(torch.float32)
    gate, up = gate_up[:intermediate], gate_up[intermediate:]
    act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).to(torch.float32)
    down = (act @ w2.t()) * route_weight
    return down.to(torch.bfloat16)


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

    w13_scale_2 / w2_scale_2 are per-expert global scales, either scalar per
    expert ([E] or [E, 1..2]) or per output row ([E, N]); the w13 pair
    (gate, up) uses columns 0 and 1 when two values are given per expert.
    """
    m, top_k = topk_ids.shape
    out = torch.zeros((m, x.shape[1]), dtype=torch.float32)
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for t in range(m):
        for r in range(top_k):
            e = int(topk_ids[t, r])
            if e < 0:
                continue
            if e not in cache:
                g13 = w13_scale_2[e]
                if g13.dim() == 1 and g13.numel() == 2:
                    n_half = w13_packed.shape[1] // 2
                    g13 = torch.cat((g13[0].repeat(n_half), g13[1].repeat(n_half)))
                cache[e] = (
                    dequantize_rows(w13_packed[e], w13_scale[e], g13),
                    dequantize_rows(w2_packed[e], w2_scale[e], w2_scale_2[e]),
                )
            w13, w2 = cache[e]
            out[t] += expert_forward(x[t], w13, w2, float(topk_weights[t, r])).to(
                torch.float32
            )
    return out.to(torch.bfloat16)
