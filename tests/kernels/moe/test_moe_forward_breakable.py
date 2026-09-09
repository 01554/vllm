# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE custom op is a breakable-CUDA-graph break point only for layers
that hold an expert cache provider; uncached layers stay in the segment."""

from types import SimpleNamespace
from unittest import mock

import torch

from vllm.compilation import breakable_cudagraph as bcg
from vllm.model_executor.layers.fused_moe.runner import moe_runner
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)


class _FakeRunner(MoERunnerInterface):
    """Passes get_layer_from_name's isinstance check; nothing else is used."""

    def __init__(self, tag: str, cached: bool):
        self.tag = tag
        self.routed_experts = SimpleNamespace(
            expert_weight_provider=object() if cached else None
        )

    def forward(self, *a, **k):  # pragma: no cover - interface stub
        raise NotImplementedError

    @property
    def shared_experts(self):  # pragma: no cover - interface stub
        return None

    @property
    def _quant_method(self):  # pragma: no cover - interface stub
        raise NotImplementedError

    def _replace_quant_method(self, quant_method):  # pragma: no cover
        raise NotImplementedError


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


def test_legacy_placeholder_consumes_one_layer_per_op(monkeypatch):
    """With the legacy "from_forward_context" name the lookup is stateful;
    the wrapper must resolve it once per op, for cached and uncached layers
    alike (the shared variant uses the same wrapper)."""
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    seen: list[str] = []

    def fn(hidden_states, router_logits, shared_experts_input, input_ids, name, pad):
        seen.append(moe_runner.get_layer_from_name(name).tag)  # plain fetch
        return hidden_states

    class Capture:
        _capturing = True
        thunks: list = []

        def add_eager(self, thunk):
            Capture.thunks.append(thunk)  # kept, as the real capture does
            return thunk()

    Capture.thunks = []
    layers = {"a": _FakeRunner("a", cached=True), "b": _FakeRunner("b", cached=False)}
    ctx = SimpleNamespace(
        all_moe_layers=["a", "b"], moe_layer_index=0, no_compile_layers=layers
    )
    with (
        mock.patch.object(moe_runner, "_USE_LAYERNAME", False),
        mock.patch.object(moe_runner, "get_forward_context", lambda: ctx),
        mock.patch.object(moe_runner, "_resolve_layer_name", lambda n: n),
        mock.patch.object(
            bcg.BreakableCUDAGraphCapture, "current", classmethod(lambda cls: Capture())
        ),
        mock.patch.object(bcg, "is_forward_context_available", lambda: False),
    ):
        wrapped = moe_runner._eager_break_when_cached(fn)
        x = torch.zeros(2, 4)
        wrapped(x, x, None, None, "from_forward_context", 0)
        wrapped(x, x, None, None, "from_forward_context", 0)
        assert seen == ["a", "b"]
        assert ctx.moe_layer_index == 2
        # Replay of the recorded (cached) thunk fetches the fixed concrete
        # name again and does not touch the forward-context index.
        assert len(Capture.thunks) == 1
        Capture.thunks[0]()
    assert seen == ["a", "b", "a"]
    assert ctx.moe_layer_index == 2


def test_identity_when_breakable_graphs_are_off(monkeypatch):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")

    def fn(*a):
        return a

    assert moe_runner._eager_break_when_cached(fn) is fn
