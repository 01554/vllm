# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loader side of the native NVFP4 backend (FreeToken GEMV, `native_nvfp4`).

The Marlin path repacks every expert bank during `process_weights_after_loading`
and folds the two w13 global scales into one. The native adapter reads the
checkpoint layout instead: packed uint8 [E, N, K/2], E4M3 block scales
[E, N, K/16], and one global scale per output row [E, N] (float16). This
module runs before the Marlin conversion and leaves the raw packed weights
and block scales untouched, expanding only the global scales (w13: gate rows
take column 0 and up rows column 1; w2: one value per row). The layout is
exclusive per process: either every layer is native or every layer is
Marlin, decided by `VLLM_LAB_EXPERT_TIER_MOE_KERNEL`.
"""

from __future__ import annotations

from typing import Any


def activation_name(activation):
    """The activation as the adapter's string: RoutedExperts holds the
    MoEActivation enum (value "silu"), tests and configs may hold a str."""
    value = getattr(activation, "value", activation)
    if not isinstance(value, str):
        raise TypeError(f"Unsupported MoE activation representation: {activation!r}")
    return value


def require_silu(activation):
    name = activation_name(activation)
    if name != "silu":
        raise NotImplementedError(
            f"Native backend supports SiLU experts only (got {name!r})"
        )
    return name


def expand_w13_globals(scale_2, intermediate, dtype=None):
    """[E, 2] -> [E, 2I]: gate rows carry column 0, up rows column 1."""
    import torch

    if scale_2.ndim != 2 or scale_2.shape[1] != 2:
        raise ValueError("w13_weight_scale_2 must be [experts, 2]")
    dtype = torch.float16 if dtype is None else dtype
    gate = scale_2[:, 0:1].expand(-1, intermediate)
    up = scale_2[:, 1:2].expand(-1, intermediate)
    return torch.cat((gate, up), dim=1).to(dtype).contiguous()


def expand_w2_globals(scale_2, hidden, dtype=None):
    """[E] -> [E, H]: one global per output row."""
    import torch

    if scale_2.ndim != 1:
        raise ValueError("w2_weight_scale_2 must be [experts]")
    dtype = torch.float16 if dtype is None else dtype
    return scale_2[:, None].expand(-1, hidden).to(dtype).contiguous()


def native_bank_shapes(w13, w2):
    """(E, hidden, intermediate) from the raw packed banks; validates layout."""
    import torch

    if w13.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise TypeError("Native backend needs the raw packed uint8 checkpoint banks")
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("Packed banks must be [experts, N, K/2]")
    experts, two_inner, half_hidden = w13.shape
    hidden, intermediate = 2 * half_hidden, two_inner // 2
    if two_inner % 2 or w2.shape != (experts, hidden, intermediate // 2):
        raise ValueError("w13 [E, 2I, H/2] and w2 [E, H, I/2] disagree")
    return experts, hidden, intermediate


def _set_parameter(layer, name, tensor):
    import torch

    if tensor is None:
        setattr(layer, name, None)
    else:
        setattr(layer, name, torch.nn.Parameter(tensor, requires_grad=False))


def prepare_native_layer(method: Any, layer: Any, replace_parameter=None):
    """Replace the Marlin conversion for one MoE layer.

    Keeps `w13_weight`, `w2_weight`, and both block scales as loaded,
    replaces both global scales with per-row float16 tensors and drops the
    input scales (BF16 activations). No kernel object is built here.
    `replace_parameter` defaults to vLLM's (reload-aware); tests pass a
    plain setter.
    """
    import torch

    if replace_parameter is None:
        from vllm.model_executor.utils import replace_parameter

    require_silu(getattr(layer, "activation", "silu"))
    if not getattr(method.moe, "is_act_and_mul", True):
        raise NotImplementedError("Native backend expects gate/up (act-and-mul)")
    experts, hidden, intermediate = native_bank_shapes(
        layer.w13_weight, layer.w2_weight
    )
    for name, shape in (
        ("w13_weight_scale", (experts, 2 * intermediate, hidden // 16)),
        ("w2_weight_scale", (experts, hidden, intermediate // 16)),
    ):
        scale = getattr(layer, name)
        if scale.dtype != torch.float8_e4m3fn or tuple(scale.shape) != shape:
            raise ValueError(f"{name}: expected E4M3 block scales {shape}")
    replace_parameter(
        layer,
        "w13_weight_scale_2",
        expand_w13_globals(layer.w13_weight_scale_2.data, intermediate),
    )
    replace_parameter(
        layer,
        "w2_weight_scale_2",
        expand_w2_globals(layer.w2_weight_scale_2.data, hidden),
    )
    replace_parameter(layer, "w13_input_scale", None)
    replace_parameter(layer, "w2_input_scale", None)
    return experts, hidden, intermediate
