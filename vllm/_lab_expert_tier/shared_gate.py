# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused shared-expert gate for the Qwen4 exp MoE block.

The shared expert's output is scaled by `sigmoid(x @ gate_weight)`, one
scalar per token. In vLLM that is a cuBLAS dot (two kernels), a sigmoid,
and a broadcast multiply per layer; FreeToken computes the gate in one
Triton program (`moe_shared_gate._gate_sigmoid_kernel`) and applies it
in another. This module goes one step further for the vLLM layout: one
program per token computes the dot product, the sigmoid, and the scaled
output, keeping vLLM's rounding points (the dot rounds to the activation
dtype, the sigmoid result rounds to it, then the multiply rounds to it),
so the result matches `F.sigmoid(gate(x)) * out` up to the dot's
accumulation order. The routed + shared addition stays in the MoE runner.

`fuse_shared_gate(mlp)` replaces the gated MLP's forward; off CUDA the
torch reference runs, and the CPU test holds it equal to the original.
"""

from __future__ import annotations

import types
from typing import Any

MAX_HIDDEN = 8192


def gated_output_reference(x, gate_weight, out):
    """`sigmoid(x @ w) * out` with vLLM's dtype rounding points."""
    import torch

    dtype = out.dtype
    logits = (x.to(torch.float32) @ gate_weight.to(torch.float32).reshape(-1)).to(dtype)
    gate = torch.sigmoid(logits.to(torch.float32)).to(dtype)
    return (gate.to(torch.float32)[:, None] * out.to(torch.float32)).to(dtype)


def gated_output(x, gate_weight, out):
    """Fused gate: Triton on CUDA, the reference elsewhere."""
    if x.device.type != "cuda":
        return gated_output_reference(x, gate_weight, out)
    import torch

    tokens, hidden = x.shape
    if hidden > MAX_HIDDEN:
        raise ValueError("Fused shared gate supports hidden sizes up to MAX_HIDDEN")
    weight = gate_weight.reshape(-1)
    if weight.shape[0] != hidden or out.shape != x.shape:
        raise ValueError("Gate weight and outputs must match the hidden size")
    x_c, out_c, w_c = x.contiguous(), out.contiguous(), weight.contiguous()
    result = torch.empty_like(out_c)
    _gate_kernel()[(tokens,)](
        x_c,
        w_c,
        out_c,
        result,
        x_c.stride(0),
        out_c.stride(0),
        HIDDEN=hidden,
        BLOCK=_next_power_of_two(hidden),
        num_warps=4 if hidden <= 2048 else 8,
    )
    return result


def fuse_shared_gate(mlp):
    """Route a gated `Qwen2MoeMLP`-style module through the fused gate.

    The module must expose `gate_up_proj`, `act_fn`, `down_proj`, and an
    `expert_gate` linear with a `weight` of shape [1, hidden] and no bias.
    """
    gate = getattr(mlp, "expert_gate", None)
    if gate is None:
        return False
    if getattr(gate, "bias", None) is not None:
        raise NotImplementedError("Fused shared gate expects a bias-free gate")
    if tuple(gate.weight.shape[:1]) != (1,):
        raise NotImplementedError("Fused shared gate expects one output feature")

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)
        return gated_output(x, self.expert_gate.weight, out)

    mlp.forward = types.MethodType(forward, mlp)
    mlp._lab_fused_shared_gate = True
    return True


def _next_power_of_two(value):
    return 1 << max(int(value) - 1, 0).bit_length()


_KERNELS: dict[str, Any] = {}


def _gate_kernel():
    if "gate" in _KERNELS:
        return _KERNELS["gate"]
    from vllm.triton_utils import tl, triton

    @triton.jit
    def shared_gate_apply(
        x_ptr,
        w_ptr,
        out_ptr,
        result_ptr,
        stride_x,
        stride_o,
        HIDDEN: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        mask = offs < HIDDEN
        x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        logit = tl.sum(x.to(tl.float32) * w.to(tl.float32), 0)
        # Round the dot and the sigmoid to the activation dtype like vLLM's
        # linear -> sigmoid -> multiply sequence does.
        logit = logit.to(x.dtype).to(tl.float32)
        gate = tl.sigmoid(logit).to(x.dtype).to(tl.float32)
        out = tl.load(out_ptr + row * stride_o + offs, mask=mask, other=0.0)
        scaled = (gate * out.to(tl.float32)).to(out.dtype)
        tl.store(result_ptr + row * stride_o + offs, scaled, mask=mask)

    _KERNELS["gate"] = shared_gate_apply
    return shared_gate_apply


__all__ = [
    "MAX_HIDDEN",
    "fuse_shared_gate",
    "gated_output",
    "gated_output_reference",
]
