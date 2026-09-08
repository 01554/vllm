# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests using real methods without CUDA model imports."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[3]


def methods(path, names, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name
        ).body
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    return namespace


@dataclass
class Config:
    speculative_config: object
    model_config: object
    quant_config: object


@pytest.mark.parametrize("start", [2, 48])
def test_mixed_fp8_lookup_uses_runtime_mtp_prefix(start):
    original = {
        "mtp.layers.0.mlp.experts": {"quant_algo": "FP8_PB_WO"},
        "model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
    }
    quant = SimpleNamespace(
        quantized_layers=original,
        ignored_layers=["mtp.layers.0.norm"],
        exclude_modules=["lm_head"],
    )
    target_quant = object()
    draft_model = object()
    config = Config(
        SimpleNamespace(draft_model_config=draft_model), object(), target_quant
    )
    ns = methods(
        "vllm/models/qwen4_exp/nvidia/mtp.py",
        {"_remap_ignored_layers", "_make_draft_vllm_config"},
        dict(
            re=re,
            replace=replace,
            Qwen4ExpMTP=object(),
            get_draft_quant_config=lambda _: quant,
            configure_quant_config=lambda *args: None,
        ),
    )
    result = ns["_make_draft_vllm_config"](config, start)
    assert result.model_config is draft_model
    assert config.quant_config is target_quant
    assert quant.quantized_layers is not original
    assert "mtp.layers.0.mlp.experts" in original
    assert quant.ignored_layers == [f"mtp.layers.{start}.norm"]
    assert quant.exclude_modules == ["lm_head"]
    lookup = methods(
        "vllm/model_executor/layers/quantization/modelopt.py",
        {"_resolve_quant_algo", "_quantized_layer_prefix_candidates"},
        {},
        "ModelOptMixedPrecisionConfig",
    )
    quant._quantized_layer_prefix_candidates = lookup[
        "_quantized_layer_prefix_candidates"
    ]
    quant.packed_modules_mapping = {}
    resolve = lookup["_resolve_quant_algo"]
    assert resolve(quant, f"mtp.layers.{start}.mlp.experts") == "FP8_PB_WO"
    assert resolve(quant, "model.layers.0.mlp.experts") == "NVFP4"


@pytest.mark.parametrize("fail", [False, True])
def test_mtp_load_scope_covers_loader_and_restores_on_error(fail):
    from vllm._lab_expert_tier.draft_scope import is_draft_load_scope

    model = SimpleNamespace(model=SimpleNamespace())

    def load(target, config):
        assert is_draft_load_scope()
        if fail:
            raise ValueError("load failed")
        return model

    ns = methods(
        "vllm/v1/worker/gpu/spec_decode/mtp/speculator.py",
        {"load_draft_model"},
        {"load_eagle_model": load},
        "MTPSpeculator",
    )
    speculator = SimpleNamespace(vllm_config=SimpleNamespace(speculative_config=None))
    assert not is_draft_load_scope()
    if fail:
        with pytest.raises(ValueError, match="load failed"):
            ns["load_draft_model"](speculator, object(), set())
    else:
        assert ns["load_draft_model"](speculator, object(), set()) is model
    assert not is_draft_load_scope()
