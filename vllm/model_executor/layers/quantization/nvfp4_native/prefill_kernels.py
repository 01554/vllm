# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# This file contains a small vLLM adapter around the grouped NVFP4 kernel from
# FreeToken.  The grouped kernel is derived from
# ``freetoken/kernel/triton/nvfp4_fused_moe.py`` at FreeToken commit
# ``af71ba43206e124f5ff6419b47ee36c6e9981078``.  FreeToken is distributed
# under the Apache License, Version 2.0; see its LICENSE file.

"""Lazy grouped Triton kernels for native NVFP4 prefill.

The grouped kernel keeps the checkpoint-native weight representation in place:
packed FP4 codes are loaded as bytes and block scales are decoded in the K
loop.  The host aligns logical expert IDs, then maps one logical expert per
aligned block to a physical bank row.  Keeping those two operations separate
is required when a bank has more physical rows than the logical expert count.
"""

from __future__ import annotations

from typing import Any

import torch

_KERNELS: dict[str, Any] = {}


def _device_index(device: torch.device) -> int:
    index = device.index
    if index is None:
        index = torch.accelerator.current_device_index()
    return index


def _triton_compute_type(dtype: torch.dtype) -> Any:
    import triton.language as tl_module

    if dtype == torch.bfloat16:
        return tl_module.bfloat16
    if dtype == torch.float16:
        return tl_module.float16
    return tl_module.float32


def _build_kernels() -> tuple[Any, Any]:
    cached = _KERNELS.get("all")
    if cached is not None:
        return cached

    import triton
    import triton.language as tl

    @triton.jit
    def map_aligned_experts_kernel(
        logical_ptr,
        map_ptr,
        physical_ptr,
        post_padded_ptr,
        error_ptr,
        num_blocks,
        block_size,
        num_experts,
        num_rows,
        WRITE_ERROR: tl.constexpr,
    ):
        offs = tl.program_id(0) * 256 + tl.arange(0, 256)
        in_bounds = offs < num_blocks
        post_padded = tl.load(post_padded_ptr)
        valid_block = in_bounds & (offs * block_size < post_padded)
        logical = tl.load(logical_ptr + offs, mask=in_bounds, other=-1).to(tl.int64)
        valid_expert = valid_block & (logical >= 0) & (logical < num_experts)
        safe = tl.where(valid_expert, logical, 0).to(tl.int64)
        mapped = tl.load(map_ptr + safe, mask=valid_expert, other=-1).to(tl.int64)
        valid_row = valid_expert & (mapped >= 0) & (mapped < num_rows)
        if WRITE_ERROR:
            bad = valid_block & ~valid_row
            bad_any = tl.sum(bad.to(tl.int32), axis=0) != 0
            tl.atomic_or(error_ptr, 1, mask=bad_any)
        result = tl.where(valid_row, mapped, -1).to(tl.int32)
        tl.store(physical_ptr + offs, result, mask=in_bounds)

    @triton.jit
    def prefill_nvfp4_moe_kernel(
        a_ptr,  # [M, K] activations
        packed_ptr,  # [S, N, K // 2] raw uint8
        scale_ptr,  # [S, N, K // 16] E4M3 bytes or native fp8
        global_ptr,  # [S, N] fp16 or fp32
        c_ptr,  # [M * top_k, N] flattened output
        topk_weights_ptr,  # [M * top_k]
        sorted_token_ids_ptr,
        physical_expert_ids_ptr,
        num_tokens_post_padded_ptr,
        lut_ptr,
        N,
        K,
        EM,
        num_rows,
        num_valid_tokens,
        stride_am,
        stride_ak,
        stride_pe,
        stride_pn,
        stride_pkb,
        stride_se,
        stride_sn,
        stride_sblk,
        stride_ge,
        stride_gn,
        stride_cm,
        stride_cn,
        stride_tw,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_KB: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        TOP_K: tl.constexpr,
        USE_NATIVE_SCALE: tl.constexpr,
        compute_type: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
        token_mask = (offs_token >= 0) & (offs_token < num_valid_tokens)

        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_kb = tl.arange(0, BLOCK_SIZE_KB)
        a_ptrs_lo = a_ptr + (
            (offs_token[:, None] // TOP_K) * stride_am
            + (2 * offs_kb)[None, :] * stride_ak
        )
        a_ptrs_hi = a_ptr + (
            (offs_token[:, None] // TOP_K) * stride_am
            + (2 * offs_kb + 1)[None, :] * stride_ak
        )

        raw_slot = tl.load(physical_expert_ids_ptr + pid_m).to(tl.int64)
        valid_slot = (raw_slot >= 0) & (raw_slot < num_rows)
        # The block map kernel has already checked the physical row; valid_slot
        # protects all loads if a stale or malformed block map is observed.
        slot = tl.where(valid_slot, raw_slot, 0).to(tl.int64)
        packed_base = packed_ptr + slot * stride_pe + offs_bn[None, :] * stride_pn
        scale_base = scale_ptr + slot * stride_se + offs_bn[None, :] * stride_sn

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        K_BYTES = K // 2
        for kb in range(0, tl.cdiv(K_BYTES, BLOCK_SIZE_KB)):
            byte_idx = kb * BLOCK_SIZE_KB + offs_kb
            byte_mask = byte_idx < K_BYTES

            p_ptrs = packed_base + byte_idx[:, None] * stride_pkb
            bytes_ = tl.load(
                p_ptrs,
                mask=valid_slot & byte_mask[:, None],
                other=0,
            ).to(tl.int32)
            lo = bytes_ & 0xF
            hi = (bytes_ >> 4) & 0xF
            sblk = byte_idx // 8
            s_ptrs = scale_base + sblk[:, None] * stride_sblk
            scale_mask = valid_slot & byte_mask[:, None]
            if USE_NATIVE_SCALE:
                scale = tl.load(s_ptrs, mask=scale_mask, other=0.0).to(tl.float32)
            else:
                bits = tl.load(s_ptrs, mask=scale_mask, other=0).to(tl.uint8)
                half_bits = ((bits & 0x80).to(tl.uint16) << 8) | (
                    (bits & 0x7F).to(tl.uint16) << 7
                )
                scale = half_bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
            b_lo = tl.load(lut_ptr + lo) * scale
            b_hi = tl.load(lut_ptr + hi) * scale

            load_mask = token_mask[:, None] & valid_slot & byte_mask[None, :]
            a_lo = tl.load(a_ptrs_lo, mask=load_mask, other=0.0)
            a_hi = tl.load(a_ptrs_hi, mask=load_mask, other=0.0)
            accumulator += tl.dot(a_lo, b_lo.to(a_lo.dtype))
            accumulator += tl.dot(a_hi, b_hi.to(a_hi.dtype))

            a_ptrs_lo += BLOCK_SIZE_KB * 2 * stride_ak
            a_ptrs_hi += BLOCK_SIZE_KB * 2 * stride_ak

        g = tl.load(
            global_ptr + slot * stride_ge + offs_bn * stride_gn,
            mask=valid_slot & (offs_bn < N),
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * g[None, :]

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(
                topk_weights_ptr + offs_token * stride_tw,
                mask=token_mask & valid_slot,
                other=0.0,
            )
            accumulator = accumulator * moe_weight[:, None]

        accumulator = accumulator.to(compute_type)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    _KERNELS["all"] = (map_aligned_experts_kernel, prefill_nvfp4_moe_kernel)
    return _KERNELS["all"]


def prefill_config(tokens: int) -> dict[str, int]:
    """Return the fixed grouped-kernel configuration used by FreeToken."""

    if tokens <= 64:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_KB": 32,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 4,
        }
    return {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_KB": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 4,
    }


def warmup(device: torch.device | str) -> None:
    """Initialize the LUT and compile the grouped kernels lazily."""

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("native prefill warmup requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("native prefill warmup requires a CUDA runtime")
    from . import kernels as decode_kernels

    with torch.accelerator.device_index(_device_index(device)):
        decode_kernels._e2m1_lut(device)
        decode_kernels._native_e4m3(device)
        decode_kernels._build_kernels()


def map_aligned_experts_inplace(
    logical_expert_ids: torch.Tensor,
    physical_expert_ids: torch.Tensor,
    step_map: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
    num_rows: int,
    error: torch.Tensor,
) -> None:
    """Map valid aligned logical blocks to physical bank rows."""

    if logical_expert_ids.device.type != "cuda":
        raise ValueError("aligned expert mapping requires CUDA tensors")
    if logical_expert_ids.dtype != torch.int32:
        raise TypeError("logical expert IDs must be int32")
    if physical_expert_ids.shape != logical_expert_ids.shape:
        raise ValueError("physical and logical expert buffers must have equal shape")
    if step_map.dtype != torch.int32 or step_map.ndim != 1:
        raise TypeError("step_map must be a one-dimensional int32 tensor")
    if (
        num_tokens_post_padded.shape != (1,)
        or num_tokens_post_padded.dtype != torch.int32
    ):
        raise TypeError("num_tokens_post_padded must be int32[1]")
    if error.shape != (1,) or error.dtype != torch.int32:
        raise TypeError("error must be int32[1]")
    if block_size <= 0 or num_experts <= 0 or num_rows <= 0:
        raise ValueError("aligned mapping dimensions must be positive")

    map_kernel, _ = _build_kernels()
    map_kernel[((logical_expert_ids.numel() + 255) // 256,)](
        logical_expert_ids,
        step_map,
        physical_expert_ids,
        num_tokens_post_padded,
        error,
        logical_expert_ids.numel(),
        block_size,
        num_experts,
        num_rows,
        WRITE_ERROR=True,
    )


def launch_prefill_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    physical_expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    num_valid_tokens: int,
    kernel_top_k: int,
    mul_routed_weight: bool,
    config: dict[str, int],
    num_rows: int,
) -> None:
    """Launch one grouped inline-dequant NVFP4 GEMM."""

    if a.device.type != "cuda":
        raise ValueError("grouped native prefill requires CUDA tensors")
    if packed.dtype != torch.uint8 or not packed.is_contiguous():
        raise TypeError("grouped native prefill requires contiguous raw uint8 weights")
    if c.ndim != 2 or not c.is_contiguous():
        raise ValueError("grouped prefill output must be contiguous [routes,N]")
    if (
        sorted_token_ids.dtype != torch.int32
        or physical_expert_ids.dtype != torch.int32
    ):
        raise TypeError("alignment buffers must be int32")
    if kernel_top_k <= 0 or num_valid_tokens < 0 or num_rows <= 0:
        raise ValueError("grouped prefill launch dimensions must be positive")

    from . import kernels as decode_kernels

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    use_native_scale = fp8_dtype is not None and block_scale.dtype == fp8_dtype
    if block_scale.dtype not in (torch.uint8, fp8_dtype):
        raise TypeError("block scales must be uint8 or float8_e4m3fn")
    if use_native_scale and not decode_kernels._native_e4m3(block_scale.device):
        block_scale = block_scale.view(torch.uint8)
        use_native_scale = False
    if not block_scale.is_contiguous():
        raise ValueError("grouped native prefill scales must be contiguous")

    _, kernel = _build_kernels()
    n = packed.shape[1]
    k = packed.shape[2] * 2
    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    grid = (
        (
            (sorted_token_ids.shape[0] + block_m - 1)
            // block_m
            * ((n + block_n - 1) // block_n)
        ),
    )
    kernel[grid](
        a,
        packed,
        block_scale,
        global_scale,
        c,
        topk_weights,
        sorted_token_ids,
        physical_expert_ids,
        num_tokens_post_padded,
        decode_kernels._e2m1_lut(a.device),
        n,
        k,
        sorted_token_ids.shape[0],
        num_rows,
        num_valid_tokens,
        a.stride(0),
        a.stride(1),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        block_scale.stride(0),
        block_scale.stride(1),
        block_scale.stride(2),
        global_scale.stride(0),
        global_scale.stride(1),
        c.stride(0),
        c.stride(1),
        topk_weights.stride(0),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_KB=config["BLOCK_SIZE_KB"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        TOP_K=kernel_top_k,
        USE_NATIVE_SCALE=use_native_scale,
        compute_type=_triton_compute_type(c.dtype),
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


__all__ = [
    "launch_prefill_gemm",
    "map_aligned_experts_inplace",
    "prefill_config",
    "warmup",
]
