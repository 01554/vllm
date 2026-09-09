# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the native experts class through the modular-kernel
contract, on the serial CPU reference path of the kernels."""

import unittest
from types import SimpleNamespace

import torch

from tests.quantization.nvfp4_native.test_native_nvfp4 import make_bank
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.nvfp4_native import bank as native
from vllm.model_executor.layers.quantization.nvfp4_native import (
    prefill as native_prefill,
)
from vllm.model_executor.layers.quantization.nvfp4_native.bank import BANK_TENSORS
from vllm.model_executor.layers.quantization.nvfp4_native.experts import (
    NativeNvFp4Experts,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)


def make_experts(gemv_rows=1, max_num_tokens=8):
    # Bypass FusedMoEExperts.__init__ (it needs full configs); the class only
    # reads experts_per_token and max_num_tokens from moe_config here.
    experts = NativeNvFp4Experts.__new__(NativeNvFp4Experts)
    experts.moe_config = SimpleNamespace(
        experts_per_token=4, max_num_tokens=max_num_tokens
    )
    experts.gemv_rows = gemv_rows
    experts._bank = None
    experts._step_map = None
    experts._decode_workspace = None
    experts._prefill_workspace = None
    return experts


class NativeExpertsTests(unittest.TestCase):
    def setUp(self):
        self.bank = make_bank()
        self.layer = SimpleNamespace(
            **{name: SimpleNamespace(data=t) for name, t in self.bank.items()}
        )
        self.weights = torch.tensor([[0.25, 0.5, 0.125, 99.0]] * 3)
        self.ids = torch.tensor([[0, 2, 0, -1]] * 3, dtype=torch.int32)
        self.x = torch.full((3, 32), 0.0625, dtype=torch.bfloat16)

    def run_apply(self, experts, rows):
        out = torch.empty((rows, 32), dtype=torch.bfloat16)
        experts.apply(
            out,
            self.x[:rows],
            self.bank["w13_weight"],
            self.bank["w2_weight"],
            self.weights[:rows],
            self.ids[:rows],
            MoEActivation.SILU,
            3,
            None,
            None,
            None,
            torch.empty(0),
            torch.empty(0),
            None,
            False,
        )
        return out

    def test_capabilities(self):
        self.assertTrue(NativeNvFp4Experts._supports_quant_scheme(kNvfp4Static, None))
        self.assertFalse(
            NativeNvFp4Experts._supports_quant_scheme(kNvfp4Static, kNvfp4Dynamic)
        )
        self.assertTrue(NativeNvFp4Experts._supports_activation(MoEActivation.SILU))
        self.assertFalse(NativeNvFp4Experts._supports_activation(MoEActivation.GELU))
        self.assertFalse(NativeNvFp4Experts._supports_no_act_and_mul())
        self.assertTrue(
            NativeNvFp4Experts._supports_parallel_config(SimpleNamespace(use_ep=False))
        )
        self.assertFalse(
            NativeNvFp4Experts._supports_parallel_config(SimpleNamespace(use_ep=True))
        )

    def test_process_weights_builds_identity_map_and_workspaces(self):
        experts = make_experts()
        experts.process_weights_after_loading(self.layer)
        self.assertEqual(experts._step_map.tolist(), [0, 1, 2])
        self.assertEqual(set(experts._bank), set(BANK_TENSORS))
        self.assertIsNotNone(experts._decode_workspace)
        self.assertIsNotNone(experts._prefill_workspace)

    def test_decode_rows_match_the_gemv_adapter(self):
        experts = make_experts(gemv_rows=1)
        experts.process_weights_after_loading(self.layer)
        out = self.run_apply(experts, 1)
        workspace = native.allocate_workspace(self.bank, 1, 4, num_experts=3)
        reference = native.gemv(
            self.x[:1],
            self.weights[:1],
            self.ids[:1],
            self.bank,
            torch.arange(3, dtype=torch.int32),
            workspace,
        )
        self.assertTrue(torch.equal(out, reference))

    def test_larger_batches_match_the_prefill_path(self):
        experts = make_experts(gemv_rows=1)
        experts.process_weights_after_loading(self.layer)
        out = self.run_apply(experts, 3)
        workspace = native_prefill.allocate_workspace(self.bank, 8, 4, num_experts=3)
        reference = native_prefill.prefill(
            self.x,
            self.weights,
            self.ids,
            self.bank,
            torch.arange(3, dtype=torch.int32),
            workspace,
        )
        self.assertTrue(torch.equal(out, reference))

    def test_router_weights_are_passed_as_fp32(self):
        # The adapters reject non-FP32 router weights; apply converts.
        experts = make_experts(gemv_rows=1)
        experts.process_weights_after_loading(self.layer)
        self.weights = self.weights.to(torch.bfloat16)
        out = self.run_apply(experts, 1)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(out.float()).all())

    def test_rejects_unsupported_calls(self):
        experts = make_experts()
        with self.assertRaises(RuntimeError):
            self.run_apply(experts, 1)
        experts.process_weights_after_loading(self.layer)
        with self.assertRaises(NotImplementedError):
            experts.apply(
                torch.empty((1, 32), dtype=torch.bfloat16),
                self.x[:1],
                self.bank["w13_weight"],
                self.bank["w2_weight"],
                self.weights[:1],
                self.ids[:1],
                MoEActivation.SILU,
                3,
                torch.arange(3),
                None,
                None,
                torch.empty(0),
                torch.empty(0),
                None,
                False,
            )
