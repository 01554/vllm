# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loader check on a synthetic shard with the checkpoint's naming."""

import tempfile
import unittest

import torch

from benchmarks.nvfp4_native.layer_loader import load_layer_bank, tensor_name

try:
    from safetensors.torch import save_file
except ImportError:  # pragma: no cover
    save_file = None

PREFIX = "model.language_model.layers.0.mlp"


@unittest.skipIf(save_file is None, "safetensors not installed")
class LayerLoaderTests(unittest.TestCase):
    def test_bank_layout_and_manifest(self):
        n, k, e = 16, 32, 3  # intermediate 16, hidden 32
        tensors = {}
        for i in range(e):
            for proj, rows, cols in (
                ("gate_proj", n, k),
                ("up_proj", n, k),
                ("down_proj", k, n),
            ):
                tensors[tensor_name(PREFIX, i, proj, "weight")] = torch.full(
                    (rows, cols // 2), 0x20 + i, dtype=torch.uint8
                )
                tensors[tensor_name(PREFIX, i, proj, "weight_scale")] = torch.ones(
                    (rows, cols // 16), dtype=torch.float8_e4m3fn
                )
                tensors[tensor_name(PREFIX, i, proj, "weight_scale_2")] = torch.tensor(
                    0.5 + i, dtype=torch.float32
                )
                tensors[tensor_name(PREFIX, i, proj, "input_scale")] = torch.tensor(
                    1.0, dtype=torch.float32
                )
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as fh:
            save_file(tensors, fh.name)
            bank, manifest = load_layer_bank(fh.name, PREFIX, e)
        self.assertEqual(tuple(bank["w13_weight"].shape), (e, 2 * n, k // 2))
        self.assertEqual(tuple(bank["w13_weight_scale"].shape), (e, 2 * n, k // 16))
        self.assertEqual(
            bank["w13_weight_scale_2"].tolist(), [[0.5 + i] * 2 for i in range(e)]
        )
        self.assertEqual(tuple(bank["w2_weight"].shape), (e, k, n // 2))
        self.assertEqual(
            bank["w2_weight_scale_2"].tolist(), [0.5 + i for i in range(e)]
        )
        # gate rows first, then up rows
        self.assertTrue(torch.all(bank["w13_weight"][1, :n] == 0x21))
        self.assertEqual(len(manifest["sources"]), e * 12)
        self.assertEqual(manifest["input_scales"]["down_proj"], [1.0] * e)
