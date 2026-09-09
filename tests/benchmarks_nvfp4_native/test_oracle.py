# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checks of the independent oracle against hand-computed values."""

import unittest

import torch

from benchmarks.nvfp4_native.oracle import dequantize_rows, moe_forward, unpack_e2m1


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
