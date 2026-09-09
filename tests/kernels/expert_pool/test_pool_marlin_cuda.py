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

SLOTS = 4  # of E=8 experts resident per layer at start; top_k=2 staging rows


def _decode(order, device):
    logits = torch.full((1, E), -10.0, device=device)
    logits[0, order[0]] = 3.0
    logits[0, order[1]] = 2.0
    return logits


def test_two_layer_pool_decode_prefill_decode_matches_the_uncached_layers(
    dist_env,  # noqa: F811
):
    """Two layers share one bank (2 * SLOTS + staging = 10 rows > E = 8), so
    every bank call takes the logical-align + physical-remap path and a miss
    on one layer can evict the other layer's row. Decode on both layers,
    then a wide batch (bank + host-view partitions), then decode again on
    the same pool; every output must match the uncached layer."""
    device = torch.accelerator.current_accelerator()
    refs, layers = [], []
    for seed_offset in (0, 1):
        params = _quantized_weights(device, seed_offset=seed_offset)
        refs.append(_make_layer(_vllm_config(0, "cached"), params))
        layers.append(
            _make_layer(_vllm_config(SLOTS, "pool"), params, host_source=True)
        )
    model = torch.nn.ModuleDict({"a": layers[0], "b": layers[1]})
    pool = install_expert_pool(model, device, max_decode_tokens=1)
    assert pool is not None
    assert pool.rows == 2 * SLOTS + TOP_K and pool.rows > E
    assert resident_per_layer(pool.tables) == [SLOTS, SLOTS]
    pls = [layer.routed_experts.expert_pool_layer for layer in layers]
    assert all(pl is not None and pl.bank_rows == pool.rows for pl in pls)
    check_global_tables(pool.tables)
    from vllm.forward_context import set_forward_context

    cfg_ref, cfg_pool = _vllm_config(0, "cached"), _vllm_config(SLOTS, "pool")

    def run(i, x, logits, n):
        with set_forward_context(None, cfg_ref, num_tokens=n):
            want = refs[i](x, logits)
        with set_forward_context(None, cfg_pool, num_tokens=n):
            got = layers[i](x, logits)
        torch.accelerator.synchronize(device)
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
        check_global_tables(pool.tables)
        assert int(pool.tables.error[0]) == 0

    x = torch.randn(1, K, dtype=torch.bfloat16, device=device)
    # Decode steps whose routes walk every expert of both layers: misses
    # promote (evicting the least recently used row of either layer) or
    # stage into the shared staging rows.
    for order in ([0, 1], [4, 5], [6, 7], [2, 3], [0, 6], [7, 1]):
        run(0, x, _decode(order, device), 1)
    for order in ([4, 5], [6, 7], [2, 6]):
        run(1, x, _decode(order, device), 1)
    assert sum(resident_per_layer(pool.tables)) == 2 * SLOTS
    # Wide batch on layer 0: resident rows from the bank, the rest through
    # the host view; every route covered exactly once.
    xb = torch.randn(8, K, dtype=torch.bfloat16, device=device)
    run(0, xb, _routing(list(range(E)), device), 8)
    assert pls[0].partition_steps == 1
    # Decode again on the same pool after the wide batch.
    for order in ([3, 4], [7, 0]):
        run(0, x, _decode(order, device), 1)
    run(1, x, _decode([0, 1], device), 1)
    assert pls[0].decode_steps == 8 and pls[1].decode_steps == 4
