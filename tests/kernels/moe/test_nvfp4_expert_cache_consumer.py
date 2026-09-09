# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 expert cache through the real Marlin consumer, on both providers.

Experts with different global scales (``*_weight_scale_2``) are swapped and
evicted across the slot map under capacity overflow; after every prepare()
the consumer's own ``g1_alphas``/``g2_alphas`` buffers must hold, per slot,
the global scale of the expert mapped there (expectations come from an
uncached layer built from the same weights), the global-scale arguments must
be the live slot buffers rather than copies, and the output must match the
uncached layer.
"""

import types

import pytest
import torch

from tests.kernels.moe.modular_kernel_tools.parallel_utils import _set_vllm_config
from tests.kernels.moe.utils import moe_quantize_weights
from vllm.config import (
    CompilationConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    is_fp4_marlin_supported,
    prepare_nvfp4_moe_layer_for_marlin,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    is_workspace_manager_initialized,
)

pytestmark = [
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required"),
    # The predicate the NVFP4 Marlin path itself uses (CUDA, capability >= 7.5);
    # the generic check_marlin_supported group-size whitelist does not cover
    # the FP4 group of 16 and would skip on supported GPUs.
    pytest.mark.skipif(
        current_platform.is_cuda() and not is_fp4_marlin_supported(),
        reason="FP4 Marlin not supported on this GPU",
    ),
]

E, K, N, TOP_K, CAPACITY, M = 8, 256, 128, 2, 4, 8


def _vllm_config(cache_size: int, provider: str) -> VllmConfig:
    cfg = VllmConfig(
        parallel_config=ParallelConfig(), compilation_config=CompilationConfig()
    )
    cfg.kernel_config.moe_backend = "marlin"
    cfg.offload_config.moe_expert_cache_size = cache_size
    cfg.offload_config.moe_expert_cache_provider = provider
    return cfg


@pytest.fixture(scope="module")
def dist_env():
    cfg = _vllm_config(0, "cached")
    _set_vllm_config(cfg, 1, rank=0, local_rank=0)
    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.accelerator.current_accelerator())
    return cfg


def _quantized_weights(device, n: int = N):
    set_random_seed(11)
    w1 = torch.randn(E, 2 * n, K, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(E, K, n, dtype=torch.bfloat16, device=device)
    # Distinct per-expert magnitudes so the global scales differ per expert.
    mag = torch.tensor([0.5 + i for i in range(E)], device=device).view(E, 1, 1)
    w1 = (w1 * mag).to(torch.bfloat16)
    w2 = (w2 * mag.flip(0)).to(torch.bfloat16)
    w1q, w1s, w1gs = moe_quantize_weights(w1, None, "nvfp4", False, None)
    w2q, w2s, w2gs = moe_quantize_weights(w2, None, "nvfp4", False, None)
    assert w1s is not None and w1gs is not None
    assert w2s is not None and w2gs is not None
    params = {
        "w13_weight": w1q,
        "w2_weight": w2q,
        "w13_weight_scale": w1s,
        "w2_weight_scale": w2s,
        "w13_weight_scale_2": (1.0 / w1gs).unsqueeze(1).expand(-1, 2).contiguous(),
        "w2_weight_scale_2": 1.0 / w2gs,
        "w13_input_scale": torch.ones((E, 2), dtype=torch.float32, device=device),
        "w2_input_scale": torch.ones(E, dtype=torch.float32, device=device),
    }
    assert torch.unique(params["w13_weight_scale_2"][:, 0]).numel() == E
    assert torch.unique(params["w2_weight_scale_2"]).numel() == E
    return params


def _make_layer(
    cfg: VllmConfig, params: dict[str, torch.Tensor], host_source: bool = False
):
    """Build the layer; with host_source the per-expert tensors are
    registered as pinned CPU tensors, the layout the loader produces when the
    cache is enabled."""
    with set_current_vllm_config(cfg):
        # Any construction error is a failure: the Marlin capability gate is
        # the module-level skip above, and the backend is pinned to "marlin".
        layer = FusedMoEFactory(
            num_experts=E,
            top_k=TOP_K,
            hidden_size=K,
            intermediate_size=N,
            params_dtype=torch.bfloat16,
            renormalize=False,
            quant_config=ModelOptNvFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=None,
                exclude_modules=[],
            ),
            tp_size=1,
            dp_size=1,
            prefix="from_forward_context",
        )
        if cfg.offload_config.moe_expert_cache_size > 0:
            # create_weights wiring: expert tensors start in pinned host memory.
            for name in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            ):
                p = getattr(layer.routed_experts, name)
                assert p.device.type == "cpu" and p.is_pinned(), name
        for name, value in params.items():
            data = value.clone()
            if host_source and name in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            ):
                data = data.cpu().pin_memory()
            layer.routed_experts.register_parameter(
                name, torch.nn.Parameter(data, requires_grad=False)
            )
        layer._quant_method.process_weights_after_loading(layer.routed_experts)
    return layer


def _routing(order: list[int], device) -> torch.Tensor:
    # Token t routes to experts (order[t], order[(t + 1) % E]); every forward
    # touches all E experts, more than the cache holds.
    logits = torch.full((M, E), -10.0, device=device)
    for t in range(M):
        logits[t, order[t % E]] = 3.0
        logits[t, order[(t + 1) % E]] = 2.0
    return logits


@pytest.mark.parametrize("n", [N, 96])  # 96 needs Marlin tile padding
def test_host_chunked_marlin_repack_matches_device_path(dist_env, n):
    """The chunked host path must produce the device path's bytes."""
    device = torch.accelerator.current_accelerator()
    params = _quantized_weights(device, n)
    g13 = params["w13_weight_scale_2"][:, 0].contiguous()

    def run(on_host: bool, chunk: int):
        layer = types.SimpleNamespace(
            num_experts=E,
            hidden_size=K,
            intermediate_size_per_partition=n,
            params_dtype=torch.bfloat16,
        )
        src = {k: v.clone() for k, v in params.items()}
        if on_host:
            for k in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
                src[k] = src[k].cpu().pin_memory()
                assert src[k].device.type == "cpu" and src[k].is_pinned()
        outs = prepare_nvfp4_moe_layer_for_marlin(
            layer,
            src["w13_weight"],
            src["w13_weight_scale"],
            g13.cpu() if on_host else g13,
            src["w2_weight"],
            src["w2_weight_scale"],
            src["w2_weight_scale_2"].cpu() if on_host else src["w2_weight_scale_2"],
            is_act_and_mul=True,
            expert_chunk=chunk,
        )
        if on_host:
            assert all(t.device.type == "cpu" and t.is_pinned() for t in outs)
        return [t.to(device) for t in outs]

    want = run(False, E)
    got = run(True, 3)  # 3 + 3 + 2 experts
    for name, a, b in zip(
        ("w13", "w13_scale", "w13_scale_2", "w2", "w2_scale", "w2_scale_2"), want, got
    ):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert torch.equal(a, b), name


@pytest.mark.parametrize("provider", ["cached", "row"])
@pytest.mark.parametrize("host_source", [False, True])
def test_scale_2_follows_slots_through_marlin(dist_env, provider, host_source):
    device = torch.accelerator.current_accelerator()
    params = _quantized_weights(device)
    ref_cfg = _vllm_config(0, provider)
    ref = _make_layer(ref_cfg, params)
    cfg = _vllm_config(CAPACITY, provider)
    layer = _make_layer(cfg, params, host_source=host_source)
    experts = layer.routed_experts
    prov = experts.expert_weight_provider
    assert prov is not None
    qm = experts.quant_method
    assert qm.moe_quant_config is not None and qm.moe_kernel is not None
    # The consumer's global-scale arguments are the live slot buffers.
    fe = qm.moe_kernel.fused_experts
    assert prov.buf_w13_scale_2 is not None and prov.buf_w2_scale_2 is not None
    assert fe.g1_alphas.data_ptr() == prov.buf_w13_scale_2.data_ptr()
    assert fe.g2_alphas.data_ptr() == prov.buf_w2_scale_2.data_ptr()
    assert experts.w13_weight_scale_2.data_ptr() == prov.buf_w13_scale_2.data_ptr()
    assert experts.w13_weight_scale.data_ptr() == prov.buf_w13_scale.data_ptr()
    # Final-representation globals of the uncached layer, indexed by expert.
    g13_final = ref.routed_experts.w13_weight_scale_2.detach().clone()
    g2_final = ref.routed_experts.w2_weight_scale_2.detach().clone()
    assert torch.unique(g13_final).numel() == E

    passes: list[int] = []
    orig_prepare = prov.prepare

    def checked_prepare(topk_ids, unique_ids=None):
        result = orig_prepare(topk_ids, unique_ids)
        torch.accelerator.synchronize(device)
        emap = result.expert_map.tolist()
        mapped = [(e, s) for e, s in enumerate(emap) if s >= 0]
        assert mapped, "a pass must expose at least one expert"
        for e, s in mapped:
            assert torch.equal(fe.g1_alphas[s], g13_final[e]), (e, s)
            assert torch.equal(fe.g2_alphas[s], g2_final[e]), (e, s)
        passes.append(len(mapped))
        return result

    prov.prepare = checked_prepare  # type: ignore[method-assign]

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    for order in (list(range(E)), list(reversed(range(E))), [3, 0, 5, 1, 7, 2, 6, 4]):
        logits = _routing(order, device)
        with set_forward_context(None, ref_cfg, num_tokens=M):
            want = ref(x, logits)
        with set_forward_context(None, cfg, num_tokens=M):
            got = layer(x, logits)
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
    assert len(passes) >= 3 * 2, passes  # every forward overflowed the cache
    if provider == "row":
        assert prov.stats()["evictions"] > 0
    else:
        assert prov.misses > CAPACITY
