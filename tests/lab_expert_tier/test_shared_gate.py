# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused shared-expert gate must reproduce the module's own forward."""

import importlib.util
import sys
import unittest
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, cast

HERE = Path(__file__).resolve().parents[2] / "vllm" / "_lab_expert_tier"
if "lab_expert_tier" not in sys.modules:
    spec = cast(
        ModuleSpec,
        importlib.util.spec_from_file_location(
            "lab_expert_tier",
            HERE / "__init__.py",
            submodule_search_locations=[str(HERE)],
        ),
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    cast(Loader, spec.loader).exec_module(package)
from lab_expert_tier import shared_gate as sg  # noqa: E402

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

Module: Any = torch.nn.Module if torch is not None else object


class Linear(Module):
    """vLLM-style linear: returns (output, bias)."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = None

    def forward(self, x):
        return F.linear(x, self.weight), None


class GatedMLP(Module):
    """Qwen2MoeMLP shape: gate_up -> silu_and_mul -> down, scaled by the gate."""

    expert_gate: Any

    def __init__(self, hidden, inner, dtype, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.gate_up_proj = Linear(
            torch.randn(2 * inner, hidden, generator=g).to(dtype)
        )
        self.down_proj = Linear(
            torch.randn(hidden, inner, generator=g).to(dtype) / inner
        )
        self.expert_gate = Linear(
            torch.randn(1, hidden, generator=g).to(dtype) / hidden
        )
        self.act_fn = lambda t: F.silu(t[:, :inner]) * t[:, inner:]

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)
        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out
        return out


@unittest.skipIf(torch is None, "CPU torch is not installed")
class SharedGateTests(unittest.TestCase):
    def test_reference_matches_the_module_forward(self):
        for dtype, tol in ((torch.float32, 1e-6), (torch.bfloat16, 2e-2)):
            mlp = GatedMLP(64, 32, dtype, seed=1)
            x = torch.randn(5, 64, generator=torch.Generator().manual_seed(2)).to(dtype)
            expected = mlp(x)
            self.assertTrue(sg.fuse_shared_gate(mlp))
            self.assertTrue(mlp._lab_fused_shared_gate)
            actual = mlp(x)
            torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)

    def test_rounding_points_follow_the_activation_dtype(self):
        x = torch.randn(3, 16).to(torch.bfloat16)
        w = torch.randn(1, 16).to(torch.bfloat16)
        out = torch.randn(3, 16).to(torch.bfloat16)
        expected = F.sigmoid(F.linear(x, w)) * out
        torch.testing.assert_close(sg.gated_output_reference(x, w, out), expected)

    def test_modules_without_a_gate_or_with_bias_are_rejected(self):
        mlp = GatedMLP(8, 4, torch.float32, seed=3)
        mlp.expert_gate = None
        self.assertFalse(sg.fuse_shared_gate(mlp))
        mlp = GatedMLP(8, 4, torch.float32, seed=3)
        mlp.expert_gate.bias = torch.zeros(1)
        with self.assertRaises(NotImplementedError):
            sg.fuse_shared_gate(mlp)


if __name__ == "__main__":
    unittest.main()
