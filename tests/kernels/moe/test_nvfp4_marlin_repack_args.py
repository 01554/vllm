# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract check: an invalid expert_chunk is rejected before any side
effect (no workspace allocation, no warning, no conversion)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_nvfp4_moe_layer_for_marlin,
)


@pytest.mark.parametrize("chunk", [0, -1])
def test_non_positive_expert_chunk_is_rejected_before_side_effects(chunk):
    layer = SimpleNamespace(
        num_experts=2,
        hidden_size=64,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )
    w13 = torch.zeros(2, 128, 32, dtype=torch.uint8)
    w2 = torch.zeros(2, 64, 32, dtype=torch.uint8)
    s13 = torch.zeros(2, 128, 4).to(torch.float8_e4m3fn)
    s2 = torch.zeros(2, 64, 4).to(torch.float8_e4m3fn)
    g = torch.ones(2)
    with pytest.raises(ValueError, match="expert_chunk"):
        prepare_nvfp4_moe_layer_for_marlin(
            layer, w13, s13, g, w2, s2, g, is_act_and_mul=True, expert_chunk=chunk
        )
    assert not hasattr(layer, "workspace")
