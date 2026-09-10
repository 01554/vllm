# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Planner semantics including inactive suffixes and cross-layer eviction."""

import importlib.util
from pathlib import Path

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_pool import tables


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("check_weights,expected", [(False, 111), (True, 135)])
def test_staging_against_cpu_reference(check_weights, expected):
    path = (
        Path(__file__).resolve().parents[3]
        / "benchmarks/kernels/expert_pool/bench_staging.py"
    )
    spec = importlib.util.spec_from_file_location("staging_bench", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.correctness([tables], check_weights) == expected
