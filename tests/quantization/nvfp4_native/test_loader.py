# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the native backend loader branch: layout kept, globals expanded."""

import importlib.util
import unittest
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

pytest.importorskip(
    "vllm.distributed", reason="needs the full vLLM environment"
)  # the package sits under vllm.model_executor, whose import pulls it in

from vllm.model_executor.layers.quantization.nvfp4_native import loader as nl

try:
    import torch
except ImportError:
    torch = None


def make_layer(experts=3, hidden=32, intermediate=16):
    """A raw ModelOpt NVFP4 MoE layer as loaded, before any conversion."""
    layer = torch.nn.Module()
    layer.activation = "silu"

    def param(tensor):
        return torch.nn.Parameter(tensor, requires_grad=False)

    layer.w13_weight = param(
        torch.full((experts, 2 * intermediate, hidden // 2), 0x22, dtype=torch.uint8)
    )
    layer.w2_weight = param(
        torch.full((experts, hidden, intermediate // 2), 0x22, dtype=torch.uint8)
    )
    layer.w13_weight_scale = param(
        torch.ones(experts, 2 * intermediate, hidden // 16).to(torch.float8_e4m3fn)
    )
    layer.w2_weight_scale = param(
        torch.ones(experts, hidden, intermediate // 16).to(torch.float8_e4m3fn)
    )
    layer.w13_weight_scale_2 = param(
        torch.tensor([[0.25, 0.5]] * experts, dtype=torch.float32)
    )
    layer.w2_weight_scale_2 = param(torch.full((experts,), 0.125, dtype=torch.float32))
    layer.w13_input_scale = param(torch.ones(experts, 2))
    layer.w2_input_scale = param(torch.ones(experts))
    return layer


@unittest.skipIf(torch is None, "CPU torch is not installed")
class NativeLoaderTests(unittest.TestCase):
    def test_globals_expand_per_row_with_gate_and_up_columns(self):
        w13 = nl.expand_w13_globals(torch.tensor([[0.25, 0.5], [2.0, 4.0]]), 3)
        self.assertEqual(w13.dtype, torch.float16)
        self.assertEqual(w13.tolist(), [[0.25] * 3 + [0.5] * 3, [2.0] * 3 + [4.0] * 3])
        w2 = nl.expand_w2_globals(torch.tensor([0.125, 1.0]), 2)
        self.assertEqual(w2.tolist(), [[0.125, 0.125], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            nl.expand_w13_globals(torch.ones(2), 3)

    def test_bank_shapes_reject_repacked_or_disagreeing_banks(self):
        layer = make_layer()
        self.assertEqual(
            nl.native_bank_shapes(layer.w13_weight, layer.w2_weight), (3, 32, 16)
        )
        with self.assertRaises(TypeError):
            nl.native_bank_shapes(layer.w13_weight.view(torch.int32), layer.w2_weight)
        with self.assertRaises(ValueError):
            nl.native_bank_shapes(layer.w13_weight, layer.w2_weight[:, :16])

    def test_prepare_keeps_raw_banks_and_marks_the_method(self):
        from vllm.model_executor.layers.quantization.nvfp4_native import (
            bank as native_nvfp4,
        )

        layer = make_layer()
        packed_ptr = layer.w13_weight.data_ptr()
        method = SimpleNamespace(
            moe=SimpleNamespace(is_act_and_mul=True),
            moe_kernel="marlin",
            moe_quant_config=1,
        )
        self.assertEqual(
            nl.prepare_native_layer(method, layer, nl._set_parameter), (3, 32, 16)
        )
        self.assertEqual(layer.w13_weight.data_ptr(), packed_ptr)
        self.assertEqual(layer.w13_weight_scale_2.shape, (3, 32))
        self.assertEqual(layer.w13_weight_scale_2.dtype, torch.float16)
        self.assertEqual(layer.w13_weight_scale_2[0, :16].tolist(), [0.25] * 16)
        self.assertEqual(layer.w13_weight_scale_2[0, 16:].tolist(), [0.5] * 16)
        self.assertEqual(layer.w2_weight_scale_2.shape, (3, 32))
        self.assertIsNone(layer.w13_input_scale)
        self.assertIsNone(layer.w2_input_scale)
        self.assertIsNone(method.moe_kernel)
        self.assertTrue(method._lab_native)
        from vllm.model_executor.layers.quantization.nvfp4_native.bank import (
            BANK_TENSORS as TENSORS,
        )

        bank = {name: getattr(layer, name).data for name in TENSORS}
        # The prepared layer is a valid adapter bank as is.
        self.assertEqual(native_nvfp4.validate_bank(bank), (3, 32, 16))

    def test_activation_accepts_the_real_enum_and_rejects_others(self):
        """RoutedExperts stores MoEActivation, not a string: the enum's
        SiLU member must pass and any other member or string must not."""
        spec = importlib.util.spec_from_file_location(
            "moe_activation",
            Path(__file__).resolve().parents[3]
            / "vllm"
            / "model_executor"
            / "layers"
            / "fused_moe"
            / "activation.py",
        )
        module = importlib.util.module_from_spec(cast(ModuleSpec, spec))
        try:
            cast(Loader, cast(ModuleSpec, spec).loader).exec_module(module)
        except ImportError as exc:  # full vllm environment only
            self.skipTest(
                f"fused_moe.activation needs the full vllm environment: {exc}"
            )
        enum = module.MoEActivation
        self.assertEqual(nl.require_silu(enum.SILU), "silu")
        self.assertEqual(nl.require_silu("silu"), "silu")
        for bad in (enum.GELU, enum.SILU_NO_MUL, enum.SWIGLUOAI, "gelu"):
            with self.assertRaises(NotImplementedError):
                nl.require_silu(bad)
        with self.assertRaises(TypeError):
            nl.require_silu(3)
        layer = make_layer()
        layer.activation = enum.SILU
        method = SimpleNamespace(
            moe=SimpleNamespace(is_act_and_mul=True),
            moe_kernel=None,
            moe_quant_config=None,
        )
        self.assertEqual(
            nl.prepare_native_layer(method, layer, nl._set_parameter), (3, 32, 16)
        )

    def test_prepare_rejects_other_activations(self):
        layer = make_layer()
        layer.activation = "gelu"
        with self.assertRaises(NotImplementedError):
            nl.prepare_native_layer(
                SimpleNamespace(moe=SimpleNamespace()), layer, nl._set_parameter
            )


if __name__ == "__main__":
    unittest.main()
