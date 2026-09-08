# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# This file contains a small vLLM adapter around the NVFP4 decode kernel from
# FreeToken.  The wide-load kernel is derived from
# ``freetoken/kernel/triton/nvfp4_fused_moe.py`` at FreeToken commit
# ``af71ba43206e124f5ff6419b47ee36c6e9981078``.  FreeToken is distributed
# under the Apache License, Version 2.0; see its LICENSE file.

"""Lazy Triton kernels for native, packed NVFP4 GEMV.

The public native backend keeps this module importable on a CPU-only install.
Triton and CUDA are imported only when a CUDA launch is requested.  The input
weight remains the checkpoint-native ``uint8 [rows, N, K // 2]`` tensor.  The
kernel takes an internal ``int32`` view of that tensor solely to issue the
same wide loads as FreeToken's production decode kernel; an already repacked
Marlin ``int32`` tensor is rejected by the Python adapter before reaching this
module.
"""

from __future__ import annotations

from typing import Any

import torch

_E2M1_VALUES = (
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

_LUTS: dict[int, torch.Tensor] = {}
_KERNELS: dict[str, Any] = {}
_AUX_KERNELS: dict[str, Any] = {}
_NATIVE_E4M3: dict[int, bool] = {}


def _device_index(device: torch.device) -> int:
    index = device.index
    if index is None:
        index = torch.accelerator.current_device_index()
    return index


def _native_e4m3(device: torch.device) -> bool:
    """Return and cache whether ``device`` can use native fp8 E4M3 pointers."""

    index = _device_index(device)
    value = _NATIVE_E4M3.get(index)
    if value is None:
        value = torch.cuda.get_device_capability(index) >= (8, 9)
        _NATIVE_E4M3[index] = value
    return value


def _e2m1_lut(device: torch.device) -> torch.Tensor:
    """Return the cached FP4 lookup table for ``device``.

    The first call should happen during model warmup.  Keeping this cache here
    means a captured forward never allocates the LUT.
    """

    index = _device_index(device)
    lut = _LUTS.get(index)
    if lut is None:
        lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)
        _LUTS[index] = lut
    return lut


def _build_kernels() -> tuple[Any, Any]:
    """Build the map and wide-load decode kernels lazily."""

    cached = _KERNELS.get("all")
    if cached is not None:
        return cached

    # Do not move these imports to module scope: CPU vLLM imports do not have
    # to install Triton or initialize a CUDA driver.
    import triton
    import triton.language as tl

    @triton.jit
    def map_routes_kernel(
        ids_ptr,
        map_ptr,
        out_ptr,
        error_ptr,
        n_routes,
        num_experts,
        num_rows,
        USE_MAP: tl.constexpr,
        WRITE_ERROR: tl.constexpr,
    ):
        offs = tl.program_id(0) * 256 + tl.arange(0, 256)
        in_bounds = offs < n_routes
        raw = tl.load(ids_ptr + offs, mask=in_bounds, other=-1).to(tl.int64)
        valid_expert = in_bounds & (raw >= 0) & (raw < num_experts)
        safe = tl.where(valid_expert, raw, 0).to(tl.int64)
        if USE_MAP:
            mapped = tl.load(map_ptr + safe, mask=valid_expert, other=-1).to(tl.int64)
        else:
            mapped = raw
        valid_row = valid_expert & (mapped >= 0) & (mapped < num_rows)
        bad_raw = in_bounds & (raw != -1) & ((raw < 0) | (raw >= num_experts))
        bad_map = in_bounds & valid_expert & ((mapped < 0) | (mapped >= num_rows))
        if WRITE_ERROR:
            bad_any = tl.sum((bad_raw | bad_map).to(tl.int32), axis=0) != 0
            tl.atomic_or(error_ptr, 1, mask=bad_any)
        result = tl.where(valid_row, mapped, -1).to(tl.int32)
        tl.store(out_ptr + offs, result, mask=in_bounds)

    @triton.jit
    def decode_nvfp4_wide_kernel(
        a_ptr,  # [M, K] activations
        packed_ptr,  # [S, N, K // 8] int32 view of raw packed uint8
        scale_ptr,  # [S, N, K // 16] E4M3 bytes or native fp8
        global_ptr,  # [S, N] fp16 or fp32
        c_ptr,  # [M, TOP_K, N]
        topk_weights_ptr,  # [M, TOP_K]
        topk_ids_ptr,  # [M, TOP_K] physical or logical ids
        route_output_ptr,  # [M, TOP_K] optional sanitized physical rows
        expert_to_row_ptr,  # [num_experts] logical -> physical, or unused
        error_ptr,  # sticky route error, or unused
        lut_ptr,  # [16] fp32
        total_routes,
        num_rows,
        num_experts,
        N,
        K,
        stride_am,
        stride_ak,
        stride_pe,
        stride_pn,
        stride_pkw,
        stride_se,
        stride_sn,
        stride_sblk,
        stride_ge,
        stride_gn,
        stride_cm,
        stride_ck,
        stride_cn,
        stride_tw_m,
        stride_tw_k,
        stride_tid_m,
        stride_tid_k,
        stride_route_m,
        stride_route_k,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_KW: tl.constexpr,
        TOP_K: tl.constexpr,
        A_ROW_IS_ROUTE: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        USE_MAP: tl.constexpr,
        WRITE_ROUTES: tl.constexpr,
        WRITE_ERROR: tl.constexpr,
        USE_NATIVE_SCALE: tl.constexpr,
        compute_type: tl.constexpr,
    ):
        route_id = tl.program_id(0)
        n_block_id = tl.program_id(1)
        route_mask = route_id < total_routes
        token_id = route_id // TOP_K
        route_k = route_id - token_id * TOP_K

        offs_n = n_block_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        n_mask = offs_n < N

        raw_id = tl.load(
            topk_ids_ptr + token_id * stride_tid_m + route_k * stride_tid_k,
            mask=route_mask,
            other=-1,
        ).to(tl.int64)
        if USE_MAP:
            valid_expert = route_mask & (raw_id >= 0) & (raw_id < num_experts)
            safe_expert = tl.where(valid_expert, raw_id, 0).to(tl.int64)
            mapped = tl.load(
                expert_to_row_ptr + safe_expert,
                mask=valid_expert,
                other=-1,
            ).to(tl.int64)
            bad_route = route_mask & (raw_id != -1) & ~valid_expert
            bad_route = bad_route | (
                route_mask & valid_expert & ((mapped < 0) | (mapped >= num_rows))
            )
            raw_slot = mapped
        else:
            raw_slot = raw_id
            bad_route = (
                route_mask
                & (raw_slot != -1)
                & ((raw_slot < 0) | (raw_slot >= num_rows))
            )
        if WRITE_ERROR:
            # One atomic per route is enough; n_block_id 0 reports the
            # route while the other output tiles only compute its value.
            tl.atomic_or(error_ptr, 1, mask=bad_route & (n_block_id == 0))
        valid_slot = route_mask & (raw_slot >= 0) & (raw_slot < num_rows)
        # Clamp before forming every bank pointer.  The validity mask is kept
        # on every load so a -1 or an out-of-range map cannot touch row zero.
        slot = tl.where(valid_slot, raw_slot, 0).to(tl.int64)

        if WRITE_ROUTES:
            # Only the first N tile owns the route result.  The logical input
            # is a separate buffer in this mode, so later N tiles can keep
            # reading it while this store publishes the physical row.
            route_offset = (
                token_id.to(tl.int64) * stride_route_m
                + route_k.to(tl.int64) * stride_route_k
            )
            route_value = tl.where(valid_slot, raw_slot, -1).to(tl.int32)
            tl.store(
                route_output_ptr + route_offset,
                route_value,
                mask=route_mask & (n_block_id == 0),
            )

        a_row = route_id if A_ROW_IS_ROUTE else token_id
        a_base = a_ptr + a_row * stride_am
        offs_kw = tl.arange(0, BLOCK_SIZE_KW)
        K_WORDS = K // 8
        partial = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)

        packed_slot = packed_ptr + slot * stride_pe
        scale_slot = scale_ptr + slot * stride_se
        for kw_start in range(0, tl.cdiv(K_WORDS, BLOCK_SIZE_KW)):
            widx = kw_start * BLOCK_SIZE_KW + offs_kw
            w_mask = widx < K_WORDS
            lane_mask = valid_slot & w_mask[:, None] & n_mask[None, :]

            word = tl.load(
                packed_slot + offs_n[None, :] * stride_pn + widx[:, None] * stride_pkw,
                mask=lane_mask,
                other=0,
            ).to(tl.int32)
            s_ptrs = (
                scale_slot
                + offs_n[None, :] * stride_sn
                + (widx[:, None] // 2) * stride_sblk
            )
            if USE_NATIVE_SCALE:
                block_scale = tl.load(
                    s_ptrs,
                    mask=lane_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                s_bits = tl.load(
                    s_ptrs,
                    mask=lane_mask,
                    other=0,
                ).to(tl.uint8)

                # E4M3 bit patterns are decoded exactly as in FreeToken's
                # e4m3_compat.py.  This representation works on pre-SM89 too
                # and avoids passing a Triton fp8 pointer to old targets.
                half_bits = ((s_bits & 0x80).to(tl.uint16) << 8) | (
                    (s_bits & 0x7F).to(tl.uint16) << 7
                )
                block_scale = (
                    half_bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
                )

            kbase = 8 * widx
            acc_word = tl.zeros((BLOCK_SIZE_KW, BLOCK_SIZE_N), dtype=tl.float32)
            for j in tl.static_range(8):
                code = (word >> (4 * j)) & 0xF
                b = tl.load(lut_ptr + code)
                a_j = tl.load(
                    a_base + (kbase + j) * stride_ak,
                    mask=valid_slot & w_mask,
                    other=0.0,
                ).to(tl.float32)
                acc_word += a_j[:, None] * b
            partial += acc_word * block_scale

        accumulator = tl.sum(partial, axis=0)
        g = tl.load(
            global_ptr + slot * stride_ge + offs_n * stride_gn,
            mask=valid_slot & n_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * g

        if MUL_ROUTED_WEIGHT:
            route_weight = tl.load(
                topk_weights_ptr + token_id * stride_tw_m + route_k * stride_tw_k,
                mask=valid_slot,
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * route_weight

        c_ptrs = c_ptr + token_id * stride_cm + route_k * stride_ck + offs_n * stride_cn
        # Mask invalid routes explicitly.  A zero multiplier would still
        # propagate NaN from a masked lane; the store mask avoids that leak.
        tl.store(
            c_ptrs,
            tl.where(valid_slot, accumulator, 0.0).to(compute_type),
            mask=route_mask & n_mask,
        )

    _KERNELS["all"] = (map_routes_kernel, decode_nvfp4_wide_kernel)
    return _KERNELS["all"]


def _build_aux_kernels() -> tuple[Any, Any]:
    """Build the fused activation and deterministic route-reduction kernels."""

    cached = _AUX_KERNELS.get("all")
    if cached is not None:
        return cached

    import triton
    import triton.language as tl

    @triton.jit
    def activation_kernel(
        gate_up_ptr,
        activated_ptr,
        intermediate,
        compute_type: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        col_blk = tl.program_id(1)
        cols = col_blk * BLOCK + tl.arange(0, BLOCK)
        mask = cols < intermediate
        gate_up_row = row * (2 * intermediate)
        gate = tl.load(gate_up_ptr + gate_up_row + cols, mask=mask, other=0.0)
        up = tl.load(
            gate_up_ptr + gate_up_row + intermediate + cols,
            mask=mask,
            other=0.0,
        )
        gate = gate.to(tl.float32)
        up = up.to(tl.float32)
        value = gate / (1.0 + tl.exp(-gate)) * up
        tl.store(
            activated_ptr + row * intermediate + cols,
            value.to(compute_type),
            mask=mask,
        )

    @triton.jit
    def sum_routes_kernel(
        down_ptr,
        out_ptr,
        tokens,
        hidden,
        TOP_K: tl.constexpr,
        compute_type: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < tokens * hidden
        token = offs // hidden
        col = offs - token * hidden
        accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
        for route in range(TOP_K):
            value = tl.load(
                down_ptr + token * TOP_K * hidden + route * hidden + col,
                mask=mask,
                other=0.0,
            )
            accumulator += value.to(tl.float32)
        tl.store(out_ptr + offs, accumulator.to(compute_type), mask=mask)

    _AUX_KERNELS["all"] = (activation_kernel, sum_routes_kernel)
    return _AUX_KERNELS["all"]


def warmup(device: torch.device | str) -> None:
    """Initialize per-device constants and lazily define all Triton kernels."""

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("native NVFP4 warmup requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("native NVFP4 warmup requires a CUDA runtime")
    with torch.accelerator.device_index(_device_index(device)):
        _e2m1_lut(device)
        _native_e4m3(device)
        _build_kernels()
        _build_aux_kernels()


def map_routes_inplace(
    ids: torch.Tensor,
    expert_to_row: torch.Tensor | None,
    out: torch.Tensor,
    *,
    num_experts: int,
    num_rows: int,
    error: torch.Tensor | None = None,
) -> None:
    """Map logical route IDs to physical rows without a temporary tensor.

    ``-1`` is the padding sentinel and is written back as ``-1`` without
    setting ``error``.  Any other invalid expert ID, or a valid expert whose
    mapped row is absent/out of range, sets the sticky device-side error flag.
    """

    if ids.device.type != "cuda":
        raise ValueError("map_routes_inplace is only the CUDA implementation")
    if ids.dtype != torch.int32 or out.dtype != torch.int32:
        raise TypeError("native routing IDs and workspace rows must be int32")
    if out.device != ids.device or out.numel() != ids.numel():
        raise ValueError("route workspace must match ids shape and device")
    if num_experts <= 0 or num_rows <= 0:
        raise ValueError("num_experts and num_rows must be positive")
    if error is not None:
        if error.device != ids.device or error.dtype != torch.int32:
            raise TypeError("route error flag must be an int32 CUDA tensor")
        if error.numel() != 1:
            raise ValueError("route error flag must contain one element")
        if not error.is_contiguous():
            raise ValueError("route error flag must be contiguous")
    if expert_to_row is None:
        map_tensor = ids
        use_map = False
    else:
        if expert_to_row.device != ids.device:
            raise ValueError("expert_to_row must be on the routing tensor's device")
        if expert_to_row.dtype not in (torch.int32, torch.int64):
            raise TypeError("expert_to_row must be int32 or int64")
        map_tensor = expert_to_row
        use_map = True
    map_kernel, _ = _build_kernels()
    n_routes = ids.numel()
    error_ptr = out if error is None else error
    map_kernel[(triton_cdiv(n_routes, 256),)](
        ids,
        map_tensor,
        out,
        error_ptr,
        n_routes,
        num_experts,
        num_rows,
        USE_MAP=use_map,
        WRITE_ERROR=error is not None,
    )


def activation_inplace(
    gate_up: torch.Tensor,
    activated: torch.Tensor,
    *,
    activation: str = "silu",
) -> None:
    """Apply the gate/up SiLU product into a preallocated activation buffer."""

    if activation != "silu":
        raise ValueError("native NVFP4 activation kernel currently supports SiLU")
    if gate_up.ndim < 2 or activated.ndim != 2:
        raise ValueError("gate_up must end in 2I and activated must be [routes, I]")
    if gate_up.shape[-1] != 2 * activated.shape[-1]:
        raise ValueError("gate_up's final dimension must be twice activated's width")
    if gate_up.numel() != activated.shape[0] * 2 * activated.shape[1]:
        raise ValueError("gate_up and activated route counts do not match")
    if gate_up.device != activated.device:
        raise ValueError("gate_up and activated must be on the same device")
    if not gate_up.is_contiguous() or not activated.is_contiguous():
        raise ValueError("gate_up and activated must be contiguous")

    intermediate = activated.shape[1]
    total = activated.numel()
    if total == 0:
        return

    if gate_up.device.type != "cuda":
        gate = gate_up.reshape(-1, 2 * intermediate)[..., :intermediate]
        up = gate_up.reshape(-1, 2 * intermediate)[..., intermediate:]
        value = gate.float() / (1.0 + torch.exp(-gate.float())) * up.float()
        activated.copy_(value.to(dtype=activated.dtype))
        return

    import triton

    activation_kernel, _ = _build_aux_kernels()
    rows = activated.shape[0]
    block = min(
        triton.next_power_of_2(intermediate),
        512 if rows < 4096 else 1024,
    )
    grid = lambda meta: (rows, triton.cdiv(intermediate, meta["BLOCK"]))
    activation_kernel[grid](
        gate_up,
        activated,
        intermediate,
        compute_type=_triton_compute_type(activated.dtype, tl_module=None),
        BLOCK=block,
        num_warps=4,
        num_stages=2 if block == 1024 else 3,
    )


def sum_routes_inplace(down: torch.Tensor, out: torch.Tensor) -> None:
    """Reduce ``[tokens, top_k, hidden]`` in route order into ``[tokens, hidden]``."""

    if down.ndim != 3 or out.ndim != 2:
        raise ValueError(
            "down must be [tokens, top_k, hidden] and out [tokens, hidden]"
        )
    tokens, top_k, hidden = down.shape
    if out.shape != (tokens, hidden):
        raise ValueError("route reduction output shape does not match down")
    if down.device != out.device:
        raise ValueError("down and out must be on the same device")
    if top_k <= 0:
        raise ValueError("route reduction requires at least one route")
    if not down.is_contiguous() or not out.is_contiguous():
        raise ValueError("down and out must be contiguous")

    if down.device.type != "cuda":
        accumulator = torch.zeros(
            (tokens, hidden), dtype=torch.float32, device=down.device
        )
        for route in range(top_k):
            accumulator += down[:, route, :].to(torch.float32)
        out.copy_(accumulator.to(dtype=out.dtype))
        return

    _, sum_kernel = _build_aux_kernels()
    block = 256
    sum_kernel[(triton_cdiv(tokens * hidden, block),)](
        down,
        out,
        tokens,
        hidden,
        TOP_K=top_k,
        compute_type=_triton_compute_type(out.dtype, tl_module=None),
        BLOCK=block,
        num_warps=4,
    )


def triton_cdiv(x: int, y: int) -> int:
    """Integer ceil division kept local so importing this module is cheap."""

    return (x + y - 1) // y


def _buffers_overlap(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Return whether two contiguous tensors cover overlapping bytes."""

    if first.device != second.device:
        return False
    first_start = first.data_ptr()
    second_start = second.data_ptr()
    first_end = first_start + first.numel() * first.element_size()
    second_end = second_start + second.numel() * second.element_size()
    return first_start < second_end and second_start < first_end


def launch_decode_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    physical_ids: torch.Tensor,
    *,
    mul_routed_weight: bool,
    a_row_is_route: bool,
    num_rows: int,
    expert_to_row: torch.Tensor | None = None,
    num_experts: int | None = None,
    error: torch.Tensor | None = None,
    write_error: bool | None = None,
    route_output: torch.Tensor | None = None,
) -> None:
    """Launch the FreeToken-style native wide-load NVFP4 GEMV.

    ``packed`` must still be raw ``uint8``.  The int32 view is made only for
    the kernel's coalesced loads and is never exposed as a bank contract.
    When ``expert_to_row`` is supplied, ``physical_ids`` contains logical
    expert IDs and the mapping is performed in this GEMM launch.  This avoids
    a separate route-map launch in decode while retaining the physical-ID
    mode used by grouped callers.
    ``route_output`` optionally receives the sanitized physical IDs.  It must
    not alias any GEMV input because every N tile reads its route and weight
    inputs while the first N tile publishes the route result.
    """

    if a.device.type != "cuda":
        raise ValueError("launch_decode_gemm requires CUDA tensors")
    if packed.dtype != torch.uint8:
        raise TypeError("native NVFP4 GEMV requires raw uint8 packed weights")
    if not packed.is_contiguous():
        raise ValueError("native NVFP4 packed weights must be contiguous")
    if packed.shape[-1] % 4:
        raise ValueError("native NVFP4 K/2 byte width must be divisible by four")
    if physical_ids.ndim != 2 or physical_ids.dtype != torch.int32:
        raise TypeError("native NVFP4 route IDs must be contiguous int32 [M, top_k]")
    if physical_ids.device != a.device or not physical_ids.is_contiguous():
        raise ValueError(
            "native NVFP4 route IDs must be contiguous on the input device"
        )
    if route_output is not None:
        if route_output.ndim != 2 or route_output.shape != physical_ids.shape:
            raise ValueError("native NVFP4 route output must match route IDs")
        if route_output.dtype != torch.int32:
            raise TypeError("native NVFP4 route output must be int32")
        if route_output.device != a.device or not route_output.is_contiguous():
            raise ValueError(
                "native NVFP4 route output must be contiguous on the input device"
            )
    if num_rows <= 0:
        raise ValueError("native NVFP4 physical row count must be positive")
    use_map = expert_to_row is not None
    if expert_to_row is not None:
        if num_experts is None or num_experts <= 0:
            raise ValueError("inline route mapping requires positive num_experts")
        if expert_to_row.ndim != 1 or expert_to_row.dtype != torch.int32:
            raise TypeError("inline route map must be one-dimensional int32")
        if expert_to_row.numel() < num_experts:
            raise ValueError("inline route map must contain every logical expert")
        if expert_to_row.device != a.device or not expert_to_row.is_contiguous():
            raise ValueError("inline route map must be contiguous on the input device")
    elif num_experts is not None:
        raise ValueError("num_experts requires an inline route map")
    if error is not None and (
        error.device != a.device
        or error.dtype != torch.int32
        or error.shape != (1,)
        or not error.is_contiguous()
    ):
        raise ValueError("route error flag must be contiguous CUDA int32[1]")
    if write_error is None:
        write_error = error is not None
    if write_error and error is None:
        raise ValueError("inline route mapping requires a sticky error flag")
    if route_output is not None and any(
        _buffers_overlap(route_output, tensor)
        for tensor in (
            a,
            packed,
            block_scale,
            global_scale,
            c,
            topk_weights,
            physical_ids,
            expert_to_row,
            error,
        )
        if tensor is not None
    ):
        raise ValueError("native NVFP4 route output must not alias GEMV inputs")
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    use_native_scale = fp8_dtype is not None and block_scale.dtype == fp8_dtype
    if block_scale.dtype not in (torch.uint8, fp8_dtype):
        raise TypeError("native E4M3 block scales must be uint8 or float8_e4m3fn")
    if use_native_scale and not _native_e4m3(block_scale.device):
        # Triton rejects an fp8 pointer in kernels compiled for pre-SM89.  Keep
        # the same kernel source and select the byte emulation branch instead.
        block_scale = block_scale.view(torch.uint8)
        use_native_scale = False
    if not block_scale.is_contiguous():
        raise ValueError("native NVFP4 block scales must be contiguous")

    _, decode_kernel = _build_kernels()
    packed_i32 = packed.view(torch.int32)
    total_routes = physical_ids.shape[0] * physical_ids.shape[1]
    n = packed.shape[1]
    k = packed.shape[2] * 2
    deep_k = k > 2048
    block_n = 8 if deep_k else 16
    block_kw = 128 if deep_k else 16
    grid = (total_routes, triton_cdiv(n, block_n))
    map_ptr = physical_ids if expert_to_row is None else expert_to_row
    error_ptr = physical_ids if error is None else error
    route_output_ptr = physical_ids if route_output is None else route_output
    route_stride_m = (
        physical_ids.stride(0) if route_output is None else route_output.stride(0)
    )
    route_stride_k = (
        physical_ids.stride(1) if route_output is None else route_output.stride(1)
    )
    decode_kernel[grid](
        a,
        packed_i32,
        block_scale,
        global_scale,
        c,
        topk_weights,
        physical_ids,
        route_output_ptr,
        map_ptr,
        error_ptr,
        _e2m1_lut(a.device),
        total_routes,
        num_rows,
        num_experts if num_experts is not None else 0,
        n,
        k,
        a.stride(0),
        a.stride(1),
        packed_i32.stride(0),
        packed_i32.stride(1),
        packed_i32.stride(2),
        block_scale.stride(0),
        block_scale.stride(1),
        block_scale.stride(2),
        global_scale.stride(0),
        global_scale.stride(1),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        topk_weights.stride(0),
        topk_weights.stride(1),
        physical_ids.stride(0),
        physical_ids.stride(1),
        route_stride_m,
        route_stride_k,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_KW=block_kw,
        TOP_K=physical_ids.shape[1],
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        USE_MAP=use_map,
        WRITE_ROUTES=route_output is not None,
        WRITE_ERROR=write_error,
        USE_NATIVE_SCALE=use_native_scale,
        compute_type=_triton_compute_type(c.dtype, tl_module=None),
        num_warps=4,
    )


def _triton_compute_type(dtype: torch.dtype, tl_module: Any) -> Any:
    """Map a torch output dtype to Triton's type without a module import."""

    if tl_module is None:
        import triton.language as tl

        tl_module = tl
    if dtype == torch.bfloat16:
        return tl_module.bfloat16
    if dtype == torch.float16:
        return tl_module.float16
    if dtype == torch.float32:
        return tl_module.float32
    raise TypeError(f"native NVFP4 GEMV does not support output dtype {dtype}")


__all__ = [
    "activation_inplace",
    "launch_decode_gemm",
    "map_routes_inplace",
    "sum_routes_inplace",
    "warmup",
]
