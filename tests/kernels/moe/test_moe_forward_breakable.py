# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE custom op is a breakable-CUDA-graph break point only for layers
that hold an expert cache provider; uncached layers stay in the segment."""

from types import SimpleNamespace
from unittest import mock

import torch

from vllm.compilation import breakable_cudagraph as bcg
from vllm.model_executor.layers.fused_moe.runner import moe_runner


def _run(monkeypatch, cached: bool):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    calls: list[str] = []

    def fn(hidden_states, router_logits, shared_experts_input, input_ids, name, pad):
        calls.append("fn")
        return hidden_states

    class Capture:
        _capturing = True

        def add_eager(self, thunk):
            calls.append("eager")
            return thunk()

    layer = SimpleNamespace(
        routed_experts=SimpleNamespace(
            expert_weight_provider=object() if cached else None
        )
    )
    with (
        mock.patch.object(moe_runner, "get_layer_from_name", lambda n: layer),
        mock.patch.object(
            bcg.BreakableCUDAGraphCapture, "current", classmethod(lambda cls: Capture())
        ),
        mock.patch.object(bcg, "is_forward_context_available", lambda: False),
    ):
        wrapped = moe_runner._eager_break_when_cached(fn)
        assert wrapped is not fn
        x = torch.zeros(2, 4)
        out = wrapped(x, x, None, None, "layer", 0)
    assert out is x
    return calls


def test_cached_layer_breaks_the_capture(monkeypatch):
    assert _run(monkeypatch, cached=True) == ["eager", "fn"]


def test_uncached_layer_stays_in_the_segment(monkeypatch):
    assert _run(monkeypatch, cached=False) == ["fn"]


def test_identity_when_breakable_graphs_are_off(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")

    def fn(*a):
        return a

    assert moe_runner._eager_break_when_cached(fn) is fn
