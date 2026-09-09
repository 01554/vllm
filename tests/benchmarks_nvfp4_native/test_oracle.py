# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checks of the independent oracle against hand-computed values."""

import unittest

import torch

from benchmarks.nvfp4_native.oracle import (
    dequantize_blocks,
    dequantize_rows,
    expert_forward,
    moe_forward,
    unpack_e2m1,
)


class OracleTests(unittest.TestCase):
    def test_nibble_order_and_codes(self):
        # byte 0x21: low nibble 1 (0.5) is element 0, high nibble 2 (1.0) is element 1
        packed = torch.tensor([[0x21, 0xF8]], dtype=torch.uint8)
        self.assertEqual(unpack_e2m1(packed).tolist(), [[0.5, 1.0, -0.0, -6.0]])

    def test_dequantize_applies_block_and_global_scales(self):
        packed = torch.full((1, 8), 0x22, dtype=torch.uint8)  # sixteen 1.0 values
        block = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn)
        out = dequantize_rows(packed, block, torch.tensor([0.25]))
        self.assertEqual(out.shape, (1, 16))
        self.assertTrue(torch.equal(out, torch.full((1, 16), 0.5)))

    def test_padding_routes_are_skipped_and_weights_applied_once(self):
        e, n, k = 2, 32, 16
        w13 = torch.full((e, n, k // 2), 0x22, dtype=torch.uint8)
        w2 = torch.full((e, k, n // 2 // 2), 0x22, dtype=torch.uint8)
        s13 = torch.ones((e, n, k // 16), dtype=torch.float8_e4m3fn)
        s2 = torch.ones((e, k, (n // 2) // 16), dtype=torch.float8_e4m3fn)
        g13 = torch.ones((e, 2))
        g2 = torch.ones((e,))
        x = torch.full((1, k), 0.5, dtype=torch.bfloat16)
        ids = torch.tensor([[0, -1]], dtype=torch.int32)
        w = torch.tensor([[0.5, 99.0]], dtype=torch.float32)
        out = moe_forward(x, ids, w, w13, s13, g13, w2, s2, g2)
        # gate = up = 8 -> silu(8)*8 ≈ 63.97 (bf16 64) ; down = 16 * 64 = 1024 ; *0.5
        self.assertEqual(out.shape, (1, k))
        self.assertTrue(
            torch.allclose(out.float(), torch.full((1, k), 512.0), rtol=0.02)
        )

    def test_global_scale_is_applied_after_the_fp32_projection(self):
        # Non-power-of-two, mixed-sign globals: folding them into the weight
        # before the matmul changes FP32 accumulation; the oracle must not.
        torch.manual_seed(0)
        n, k = 32, 64
        w13 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
        s13 = (torch.rand((n, k // 16)) * 3 + 0.1).to(torch.float8_e4m3fn)
        w2 = torch.randint(0, 256, (k, n // 2 // 2), dtype=torch.uint8)
        s2 = (torch.rand((k, (n // 2) // 16)) * 3 + 0.1).to(torch.float8_e4m3fn)
        g13 = torch.tensor([-0.37, 1.93])
        g2 = torch.tensor([0.71])
        x = (torch.randn(k) * 0.3).to(torch.bfloat16)
        w13_b = dequantize_blocks(w13, s13)
        w2_b = dequantize_blocks(w2, s2)
        g13_rows = torch.cat((g13[0].repeat(n // 2), g13[1].repeat(n // 2)))
        g2_rows = g2.repeat(k)
        got = expert_forward(x, w13_b, g13_rows, w2_b, g2_rows, 0.5)
        # Hand-ordered expectation with the same sequence of roundings.
        gu = ((x.float() @ w13_b.t()) * g13_rows).to(torch.bfloat16).float()
        a = (
            (torch.nn.functional.silu(gu[: n // 2]) * gu[n // 2 :])
            .to(torch.bfloat16)
            .float()
        )
        want = ((a @ w2_b.t()) * g2_rows * 0.5).to(torch.bfloat16)
        self.assertTrue(torch.equal(got, want))
