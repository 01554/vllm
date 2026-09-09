# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native layout, rounding and routing contracts without a CUDA dependency."""

import unittest

import torch

from tests.quantization.nvfp4_native.native_nvfp4_smoke import native, oracle


def make_bank():
    bank = {}
    for prefix, n, k in (("w13", 32, 32), ("w2", 32, 16)):
        bank[f"{prefix}_weight"] = torch.full((3, n, k // 2), 0x22, dtype=torch.uint8)
        bank[f"{prefix}_weight_scale"] = torch.ones((3, n, k // 16)).to(
            torch.float8_e4m3fn
        )
        bank[f"{prefix}_weight_scale_2"] = torch.full(
            (3, n), 0.125, dtype=torch.float16
        )
    bank["w13_weight_scale_2"][:, :16] = 0.25
    return bank


class NativeNVFP4Tests(unittest.TestCase):
    def setUp(self):
        self.bank = make_bank()
        self.workspace = native.allocate_workspace(self.bank, 3, 4, num_experts=3)
        self.x = torch.full((2, 32), 0.0625, dtype=torch.bfloat16)
        self.weights = torch.tensor([[0.25, 0.5, 0.125, 99.0]] * 2)
        self.ids = torch.tensor([[0, 2, 0, -1]] * 2, dtype=torch.int32)
        self.mapping = torch.tensor([2, 0, 1, -1], dtype=torch.int32)

    def run_native(self):
        return native.gemv(
            self.x, self.weights, self.ids, self.bank, self.mapping, self.workspace
        )

    def run_decode_with_ready_routes(self):
        return native.gemv(
            self.x,
            self.weights,
            self.ids,
            self.bank,
            self.mapping,
            self.workspace,
            routes_ready=True,
        )

    def test_per_row_globals_router_weights_and_duplicate_routes(self):
        actual = self.run_native()
        expected = oracle(self.x, self.weights, self.ids, self.bank, self.mapping)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertGreater(actual.abs().max().item(), 0.01)
        self.assertEqual(self.workspace.error.item(), 0)
        # Duplicates are separate weighted routes, not a deduplicated GEMM sum.
        self.weights[:, 2] = 0
        without_duplicate = self.run_native().clone()
        self.assertLess(
            without_duplicate.abs().max().item(), expected.abs().max().item()
        )

    def test_padding_overwrites_previous_output_even_with_nan_input(self):
        self.run_native()
        self.ids.fill_(-1)
        self.x.fill_(float("nan"))
        self.weights.fill_(float("nan"))
        self.assertTrue(torch.equal(self.run_native(), torch.zeros_like(self.x)))
        self.assertEqual(self.workspace.error.item(), 0)

    def test_low_nibble_even_and_block_scale_boundary(self):
        # Each pair is +1, -0.5; only the second 16-element block is doubled.
        self.bank["w13_weight"].fill_(0x92)
        scales = torch.ones((3, 32, 2))
        scales[:, :, 1] = 2
        self.bank["w13_weight_scale"].copy_(scales.to(torch.float8_e4m3fn))
        self.x[:, 0::2] = 0.125
        self.x[:, 1::2] = 0.0625
        self.run_native()
        # 8*(.125-.5*.0625)*(1+2) = 2.25 before per-row globals.
        expected_gate = torch.full((16,), 2.25 * 0.25).bfloat16()
        expected_up = torch.full((16,), 2.25 * 0.125).bfloat16()
        self.assertTrue(torch.equal(self.workspace.gate_up[0, 0, :16], expected_gate))
        self.assertTrue(torch.equal(self.workspace.gate_up[0, 0, 16:], expected_up))

    def test_invalid_owner_and_ids_zero_lanes_and_set_sticky_error(self):
        self.ids.copy_(torch.tensor([[3, -2, 1, -1]] * 2))
        self.mapping[1] = -1
        self.assertTrue(torch.equal(self.run_native(), torch.zeros_like(self.x)))
        self.assertEqual(self.workspace.error.item(), 1)
        self.ids.fill_(-1)
        self.run_native()
        self.assertEqual(self.workspace.error.item(), 1)

    def test_map_and_weight_updates_are_observed_with_reused_workspace(self):
        before = self.run_native().clone()
        self.mapping[:3].copy_(torch.tensor([0, 1, 2]))
        self.bank["w2_weight"][0].zero_()
        after = self.run_native().clone()
        self.assertFalse(torch.equal(before, after))
        torch.testing.assert_close(
            after,
            oracle(self.x, self.weights, self.ids, self.bank, self.mapping),
            rtol=0,
            atol=0,
        )

    def test_routes_ready_uses_physical_rows_over_logical_inputs(self):
        physical = torch.tensor([[1, -1, 0, 2], [2, 1, -1, 0]], dtype=torch.int32)
        self.workspace.routes[:2].copy_(physical)
        row_scales = torch.tensor([0.125, 0.25, 0.5], dtype=torch.float16)
        self.bank["w2_weight_scale_2"].copy_(
            row_scales[:, None].expand_as(self.bank["w2_weight_scale_2"])
        )
        # These logical values and map entries must be ignored in ready mode.
        self.ids.copy_(torch.tensor([[-2, 99, -1, 0], [99, -2, 1, -1]]))
        self.mapping.fill_(-1)

        actual = self.run_decode_with_ready_routes()
        identity = torch.arange(3, dtype=torch.int32)
        expected = oracle(self.x, self.weights, physical, self.bank, identity)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(self.workspace.error.item(), 0)

    def test_routes_ready_invalid_rows_are_zero_and_error_reset_is_explicit(self):
        self.workspace.routes[:2].copy_(
            torch.tensor([[0, -1, 3, -2], [1, -1, -1, 0]], dtype=torch.int32)
        )
        actual = self.run_decode_with_ready_routes()
        sanitized = torch.tensor([[0, -1, -1, -1], [1, -1, -1, 0]], dtype=torch.int32)
        identity = torch.arange(3, dtype=torch.int32)
        expected = oracle(self.x, self.weights, sanitized, self.bank, identity)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(self.workspace.error.item(), 1)

        # The error is sticky across a padding-only call.
        self.workspace.routes[:2].fill_(-1)
        self.x.fill_(float("nan"))
        self.weights.fill_(float("nan"))
        self.assertTrue(
            torch.equal(self.run_decode_with_ready_routes(), torch.zeros_like(self.x))
        )
        self.assertEqual(self.workspace.error.item(), 1)

        self.workspace.error.zero_()
        self.assertTrue(
            torch.equal(self.run_decode_with_ready_routes(), torch.zeros_like(self.x))
        )
        self.assertEqual(self.workspace.error.item(), 0)

    def test_repacked_marlin_and_non_e4m3_scales_are_rejected(self):
        for name, replacement in (
            ("w13_weight", self.bank["w13_weight"].view(torch.int32)),
            ("w13_weight_scale", self.bank["w13_weight_scale"].view(torch.uint8)),
        ):
            with self.subTest(name=name):
                altered = dict(self.bank)
                altered[name] = replacement
                with self.assertRaises(TypeError):
                    native.allocate_workspace(altered, 1, 4)

    def test_projection_global_cannot_replace_per_row_globals(self):
        self.bank["w13_weight_scale_2"] = torch.ones(3)
        with self.assertRaises(ValueError):
            native.allocate_workspace(self.bank, 1, 4)

    def test_token_capacity_and_unsupported_activation_fail_before_launch(self):
        self.x = torch.zeros((4, 32), dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            self.run_native()
        with self.assertRaises(NotImplementedError):
            native.gemv(
                self.x,
                self.weights,
                self.ids,
                self.bank,
                self.mapping,
                self.workspace,
                activation="gelu",
            )


class NativePrefillTests(NativeNVFP4Tests):
    """The prefill entrypoint preserves native layout/routing contracts.

    CPU arithmetic is a fallback; the --grouped GPU smoke checks the new kernel.
    """

    def setUp(self):
        from vllm.model_executor.layers.quantization.nvfp4_native.prefill import (
            allocate_workspace,
        )

        super().setUp()
        self.workspace = allocate_workspace(self.bank, 3, 4, num_experts=3)

    def run_native(self):
        from vllm.model_executor.layers.quantization.nvfp4_native.prefill import prefill

        return prefill(
            self.x, self.weights, self.ids, self.bank, self.mapping, self.workspace
        )

    def test_prefill_matches_decode_and_reuses_own_output(self):
        expected = self.run_native().clone()
        decode_workspace = native.allocate_workspace(self.bank, 3, 4, num_experts=3)
        actual = native.gemv(
            self.x, self.weights, self.ids, self.bank, self.mapping, decode_workspace
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.ids.fill_(-1)
        self.assertTrue(torch.equal(self.run_native(), torch.zeros_like(self.x)))


if __name__ == "__main__":
    unittest.main()
