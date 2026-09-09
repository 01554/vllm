# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-level installation of the global expert pool.

Runs once after every layer's process_weights_after_loading: the MoE layers
that asked for ``moe_expert_cache_provider=pool`` keep their expert tensors
in pinned host memory (final kernel layout); this allocates one VRAM bank
shared by all of them, fills each layer's initial rows, and binds a
consumer (Marlin) to the bank.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.expert_pool.copy import configure_copy
from vllm.model_executor.layers.fused_moe.expert_pool.layer import PoolLayer
from vllm.model_executor.layers.fused_moe.expert_pool.pool import GlobalPool
from vllm.model_executor.layers.fused_moe.expert_pool.tables import (
    TENSORS,
    allocate_step_buffers,
    set_control,
)

logger = init_logger(__name__)


def _next_power_of_two(value: int) -> int:
    return 1 << max(int(value) - 1, 0).bit_length()


def pool_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if getattr(module, "expert_pool_pending", False)
    ]


def _sources(name: str, layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    sources = {}
    for tensor_name in TENSORS:
        parameter = getattr(layer, tensor_name, None)
        if parameter is None:
            raise RuntimeError(f"{name}.{tensor_name}: missing for the expert pool")
        t = parameter.data
        if (
            t.device.type != "cpu"
            or not t.is_pinned()
            or not t.is_contiguous()
            or t.ndim < 1
            or t.shape[0] != layer.local_num_experts
        ):
            raise RuntimeError(
                f"{name}.{tensor_name}: the expert pool needs a pinned, "
                "contiguous host source with one row per expert"
            )
        sources[tensor_name] = t
    return sources


def install_expert_pool(
    model: torch.nn.Module, device: torch.device, max_decode_tokens: int = 1
) -> GlobalPool | None:
    """Allocate the shared bank and bind every pending pool layer to it."""
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        make_nvfp4_moe_kernel,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    layers = pool_layers(model)
    if not layers:
        return None
    first_name, first = layers[0]
    num_experts = first.local_num_experts
    top_k = first.moe_config.experts_per_token
    slots = min(first._moe_expert_cache_size, num_experts - 1)
    if slots < top_k:
        raise ValueError(
            f"expert pool needs at least top_k={top_k} rows per layer, got {slots}"
        )
    staging = top_k * max(1, max_decode_tokens)
    width = _next_power_of_two(staging)
    sources = [_sources(name, layer) for name, layer in layers]
    for name, src in zip((n for n, _ in layers), sources):
        for tensor_name in TENSORS:
            if (
                src[tensor_name].shape[1:] != sources[0][tensor_name].shape[1:]
                or src[tensor_name].dtype != sources[0][tensor_name].dtype
            ):
                raise RuntimeError(
                    f"{name}.{tensor_name}: layer rows differ from {first_name}"
                )
    pool = GlobalPool(device, sources[0], [slots] * len(layers), staging)
    configure_copy("chunks")
    for index, ((name, layer), src) in enumerate(zip(layers, sources)):
        method = layer.quant_method
        start = pool.offset(index)
        for tensor_name in TENSORS:
            pool.bank[tensor_name][start : start + slots].copy_(
                src[tensor_name][:slots], non_blocking=True
            )
        host_views = {
            tensor_name: get_accelerator_view_from_cpu_tensor(t)
            for tensor_name, t in src.items()
        }
        proxy = SimpleNamespace(
            **{tensor_name: pool.bank[tensor_name] for tensor_name in TENSORS},
            w13_input_scale=None,
            w2_input_scale=None,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            swiglu_alpha=getattr(layer, "swiglu_alpha", None),
            swiglu_beta=getattr(layer, "swiglu_beta", None),
        )
        quant = method.get_fused_moe_quant_config(proxy)
        config = replace(method.moe, num_local_experts=pool.rows)
        kernel = make_nvfp4_moe_kernel(
            quant,
            config,
            method.experts_cls,
            method.nvfp4_backend,
            routing_tables=None,
        )
        if kernel.prepare_finalize.supports_async():
            raise NotImplementedError(
                "expert pool requires synchronous prepare/finalize"
            )
        layer.expert_pool_layer = PoolLayer(
            index=index,
            pool=pool,
            slots=slots,
            sources=src,
            host_views=host_views,
            buffers=allocate_step_buffers(device, num_experts, width),
            experts=kernel.fused_experts,
            marlin_workspace=marlin_make_workspace_new(device, 4),
            num_experts=num_experts,
            top_k=top_k,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )
        layer.expert_pool_pending = False
    # Placement policy: promotions on every forward, first miss promotes,
    # no protection window, gate open (the lab run's values).
    set_control(
        pool.tables,
        promote_limit=0,
        promote_interval=1,
        promote_min_misses=1,
        protect_recent=0,
        gate=1,
    )
    torch.accelerator.synchronize(device)
    model.expert_pool = pool
    logger.info(
        "Expert pool installed: %d layers, %d/%d rows per layer resident, "
        "%d staging rows, bank %.1f GiB (%s)",
        len(layers),
        slots,
        num_experts,
        staging,
        (pool.pool_bytes + pool.staging_bytes) / 2**30,
        type(layers[0][1].quant_method).__name__,
    )
    return pool
