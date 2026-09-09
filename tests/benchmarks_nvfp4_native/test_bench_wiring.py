# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the bench wiring that do not need CUDA."""

import inspect
import unittest

import torch


class BenchWiringTests(unittest.TestCase):
    def test_experts_apply_call_uses_the_full_positional_signature(self):
        from vllm.model_executor.layers.quantization.nvfp4_native.experts import (
            NativeNvFp4Experts,
        )

        params = [
            p
            for p in inspect.signature(NativeNvFp4Experts.apply).parameters
            if p != "self"
        ]
        self.assertEqual(len(params), 15)
        # The kernel path (KernelRunner) reaches apply() with exactly these.
        self.assertEqual(
            params,
            [
                "output",
                "hidden_states",
                "w1",
                "w2",
                "topk_weights",
                "topk_ids",
                "activation",
                "global_num_experts",
                "expert_map",
                "a1q_scale",
                "a2_scale",
                "workspace13",
                "workspace2",
                "expert_tokens_meta",
                "apply_router_weight_on_input",
            ],
        )

    def test_compare_reports_violations_and_rms(self):
        from benchmarks.nvfp4_native.bench import _compare

        got = torch.tensor([1.0, 2.0, 3.0])
        ref = torch.tensor([1.0, 2.0, 3.5])
        c = _compare(got, ref, atol=0.0002, rtol=0.03)
        self.assertEqual(c["violations"], 1)
        self.assertFalse(c["pass"])
        self.assertGreater(c["normalized_rms"], 0.0)
