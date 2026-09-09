# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The pool provider through the real Marlin consumer (CUDA).

A layer with ``moe_expert_cache_provider=pool`` and a small resident count:
decode steps (one token) must match the uncached layer while experts are
promoted and evicted through the shared bank (bank rows > expert count, so
the logical-align + physical-remap path is exercised), and a wider batch
must match through the bank + host-view partition path. The pool tables
must stay consistent throughout."""

import pytest
import torch

from tests.kernels.moe.test_nvfp4_expert_cache_consumer import (
    TOP_K,
    E,
    K,
    _make_layer,
    _quantized_weights,
    _routing,
    _vllm_config,
    dist_env,  # noqa: F401
)
from vllm.model_executor.layers.fused_moe.expert_pool.install import (
    install_expert_pool,
)
from vllm.model_executor.layers.fused_moe.expert_pool.tables import (
    check_global_tables,
    resident_per_layer,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    is_fp4_marlin_supported,
)
from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required"),
    pytest.mark.skipif(
        current_platform.is_cuda() and not is_fp4_marlin_supported(),
        reason="FP4 Marlin not supported on this GPU",
    ),
]

SLOTS = 4  # of E=8 experts resident at start; top_k=2 staging rows


def test_pool_decode_and_partition_match_the_uncached_layer(dist_env):  # noqa: F811
    device = torch.accelerator.current_accelerator()
    params = _quantized_weights(device)
    ref = _make_layer(_vllm_config(0, "cached"), params)
    layer = _make_layer(_vllm_config(SLOTS, "pool"), params, host_source=True)
    experts = layer.routed_experts
    assert experts.expert_pool_pending and experts.expert_pool_layer is None
    pool = install_expert_pool(layer, device, max_decode_tokens=1)
    assert pool is not None and experts.expert_pool_layer is not None
    assert pool.rows == SLOTS + TOP_K and resident_per_layer(pool.tables) == [SLOTS]
    check_global_tables(pool.tables)
    from vllm.forward_context import set_forward_context

    x = torch.randn(1, K, dtype=torch.bfloat16, device=device)
    # Decode steps whose routes walk every expert: misses promote (evicting
    # the least recently used row) or stage; outputs must match the
    # uncached layer each time.
    for order in ([0, 1], [4, 5], [6, 7], [2, 3], [0, 6], [7, 1]):
        logits = torch.full((1, E), -10.0, device=device)
        logits[0, order[0]] = 3.0
        logits[0, order[1]] = 2.0
        with set_forward_context(None, _vllm_config(0, "cached"), num_tokens=1):
            want = ref(x, logits)
        with set_forward_context(None, _vllm_config(SLOTS, "pool"), num_tokens=1):
            got = layer(x, logits)
        torch.accelerator.synchronize(device)
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
        check_global_tables(pool.tables)
    pl = experts.expert_pool_layer
    assert pl.decode_steps == 6 and pl.partition_steps == 0
    assert int(pool.tables.error[0]) == 0
    # Wider batch: bank partition for resident experts plus the host view for
    # the rest, all routes covered exactly once.
    xb = torch.randn(8, K, dtype=torch.bfloat16, device=device)
    logits = _routing(list(range(E)), device)
    with set_forward_context(None, _vllm_config(0, "cached"), num_tokens=8):
        want = ref(xb, logits)
    with set_forward_context(None, _vllm_config(SLOTS, "pool"), num_tokens=8):
        got = layer(xb, logits)
    torch.accelerator.synchronize(device)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
    assert pl.partition_steps == 1
    check_global_tables(pool.tables)
