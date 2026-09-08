# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests using real methods without CUDA model imports."""

import ast
import sys
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


@pytest.mark.parametrize("algo", ["FP8_PB_WO", "FP8_BLOCK_SCALES"])
@pytest.mark.parametrize("start", [2, 48])
def test_mixed_fp8_lookup_uses_runtime_mtp_prefix(start, algo, monkeypatch):
    original = {
        "mtp.layers.0.mlp.experts": {"quant_algo": algo, "group_size": 128},
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
    assert resolve(quant, f"mtp.layers.{start}.mlp.experts") == algo
    assert resolve(quant, "model.layers.0.mlp.experts") == "NVFP4"

    # Reproduce the constructor dispatch that previously returned None and
    # selected UnquantizedFusedMoEMethod despite the block-FP8 checkpoint.
    class Experts:
        moe_config = SimpleNamespace(moe_backend="marlin")

    class OtherLayer:
        pass

    config_ns = methods(
        "vllm/model_executor/layers/quantization/fp8.py",
        {"__init__"},
        {
            "super": lambda: SimpleNamespace(__init__=lambda: None),
            "ACTIVATION_SCHEMES": ["static", "dynamic"],
        },
        "Fp8Config",
    )
    block_config_cls = type("BlockConfig", (), {"__init__": config_ns["__init__"]})
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.fp8",
        SimpleNamespace(
            Fp8Config=block_config_cls,
            Fp8MoEMethod=lambda config, layer: (config, layer),
        ),
    )
    dispatch = methods(
        "vllm/model_executor/layers/quantization/modelopt.py",
        {"get_quant_method", "has_blocked_weights"},
        dict(
            Attention=OtherLayer,
            LinearBase=OtherLayer,
            ParallelLMHead=OtherLayer,
            RoutedExperts=Experts,
        ),
        "ModelOptMixedPrecisionConfig",
    )
    quant._resolve_quant_algo = lambda prefix: resolve(quant, prefix)
    quant.is_layer_excluded = lambda prefix: False
    layer = Experts()
    block_config, selected_layer = dispatch["get_quant_method"](
        quant, layer, f"mtp.layers.{start}.mlp.experts"
    )
    assert selected_layer is layer
    assert layer.moe_config.moe_backend == "marlin"
    assert block_config.is_checkpoint_fp8_serialized
    assert block_config.weight_block_size == [128, 128]
    assert block_config.activation_scheme == "dynamic"
    assert dispatch["has_blocked_weights"](quant)


@pytest.mark.parametrize("fail", [False, True])
def test_mtp_load_scope_covers_loader_and_restores_on_error(fail, monkeypatch):
    from vllm._lab_expert_tier.draft_scope import is_draft_load_scope

    model = SimpleNamespace(model=SimpleNamespace())
    target_model = object()
    recorded = []

    def record(target, draft):
        assert not is_draft_load_scope()
        assert target is target_model and draft is model
        recorded.append(draft)

    monkeypatch.setitem(
        sys.modules,
        "vllm._lab_expert_tier.runtime",
        SimpleNamespace(record_draft_model=record),
    )

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
            ns["load_draft_model"](speculator, target_model, set())
    else:
        assert ns["load_draft_model"](speculator, target_model, set()) is model
    assert recorded == ([] if fail else [model])
    assert not is_draft_load_scope()
