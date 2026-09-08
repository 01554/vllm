# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft reservation comes from checkpoint headers; resident bytes count
unique storage and report, never charge, storage shared with the target."""

import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

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
from lab_expert_tier import draft_capacity as dc  # noqa: E402
from lab_expert_tier import runtime as rt  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def write_shard(path, tensors):
    """Minimal safetensors file: header length, header JSON, raw bytes."""
    header, offset, payload = {}, 0, b""
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        payload += b"\x00" * nbytes
    raw = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)) + raw + payload)


class EstimateTests(unittest.TestCase):
    def make_checkpoint(self):
        root = tempfile.mkdtemp()
        write_shard(
            os.path.join(root, "a.safetensors"),
            {
                "mtp.layers.0.w": ("F8_E4M3", [4, 8], 32),
                "model.layers.0.w": ("BF16", [4, 4], 32),
            },
        )
        write_shard(
            os.path.join(root, "b.safetensors"),
            {"mtp.layers.0.s": ("BF16", [2], 4), "mtp.norm": ("BF16", [2], 4)},
        )
        index = {
            "weight_map": {
                "mtp.layers.0.w": "a.safetensors",
                "model.layers.0.w": "a.safetensors",
                "mtp.layers.0.s": "b.safetensors",
                "mtp.norm": "b.safetensors",
            }
        }
        with open(os.path.join(root, dc.INDEX_FILE), "w") as handle:
            json.dump(index, handle)
        return root

    def test_estimate_sums_prefixed_tensors_from_headers_only(self):
        root = self.make_checkpoint()
        estimate = dc.estimate_draft_bytes(root, "mtp.")
        self.assertEqual(estimate["bytes"], 40)
        self.assertEqual(estimate["tensors"], 3)
        self.assertEqual(estimate["by_dtype"], {"BF16": 8, "F8_E4M3": 32})
        self.assertEqual(estimate["shards"], ["a.safetensors", "b.safetensors"])
        with self.assertRaises(ValueError):
            dc.estimate_draft_bytes(root, "draft.")
        with self.assertRaises(FileNotFoundError):
            dc.estimate_draft_bytes(os.path.join(root, "missing"), "mtp.")

    def test_check_estimate_allows_tolerance_and_reports_shortfall(self):
        estimate = {"bytes": 1000}
        self.assertEqual(
            dc.check_estimate(estimate, {"unique_bytes": 1040}, 0.05),
            {"reserved_bytes": 1000, "excess_bytes": 40},
        )
        self.assertEqual(
            dc.check_estimate(estimate, {"unique_bytes": 900}, 0.0)["excess_bytes"],
            -100,
        )
        with self.assertRaises(RuntimeError):
            dc.check_estimate(estimate, {"unique_bytes": 1051}, 0.05)


@unittest.skipIf(torch is None, "torch required")
class ResidentTests(unittest.TestCase):
    def test_shared_storage_is_reported_not_charged(self):
        import torch.nn as nn

        target = nn.Module()
        target.embed = nn.Parameter(torch.zeros(4, 4))
        draft = nn.Module()
        draft.embed = target.embed  # shared with the target
        draft.own = nn.Parameter(torch.zeros(2, 4))
        draft.register_buffer("scale", torch.zeros(2))
        with patch.object(
            torch.Tensor, "device", property(lambda t: torch.device("meta"))
        ):
            measured = dc.measure_resident_bytes(draft, shared_with=target)
        self.assertEqual(measured["unique_bytes"], 8 * 4 + 2 * 4)
        self.assertEqual(measured["shared_bytes"], 16 * 4)
        self.assertEqual(measured["storages"], 2)
        host = dc.measure_resident_bytes(draft, shared_with=target)
        self.assertEqual((host["unique_bytes"], host["host_bytes"]), (0, 104))


class RuntimeSeamTests(unittest.TestCase):
    def test_reserve_only_for_draft_methods_and_budget_check(self):
        settings = rt.Settings(32 * 2**30, vram_budget_gib=1.0)
        self.assertIsNone(rt.reserve_draft(None, settings, 10))
        ngram = SimpleNamespace(method="ngram")
        self.assertIsNone(rt.reserve_draft(ngram, settings, 10))
        mtp = SimpleNamespace(
            method="mtp", draft_model_config=SimpleNamespace(model="/ckpt")
        )
        fake = {"bytes": 2**29, "tensors": 1, "by_dtype": {}, "shards": []}
        with patch.object(dc, "estimate_draft_bytes", return_value=dict(fake)):
            estimate = rt.reserve_draft(mtp, settings, 2**29)
            self.assertEqual(estimate["checkpoint"], "/ckpt")
            self.assertEqual(estimate["budget_bytes"], 2**30)
            with self.assertRaises(RuntimeError):
                rt.reserve_draft(mtp, settings, 2**29 + 1)
            no_budget = rt.Settings(32 * 2**30)
            self.assertIsNone(rt.reserve_draft(mtp, no_budget, 2**40)["budget_bytes"])

    def test_record_draft_model_is_noop_without_tier_and_needs_reserve(self):
        self.assertIsNone(rt.record_draft_model(SimpleNamespace(), SimpleNamespace()))
        coordinator = SimpleNamespace(draft_reserve=None, settings=rt.Settings(1))
        target = SimpleNamespace(_lab_expert_tier_coordinator=coordinator)
        with (
            patch.object(dc, "measure_resident_bytes", return_value={}),
            self.assertRaises(RuntimeError),
        ):
            rt.record_draft_model(target, SimpleNamespace())
        coordinator.draft_reserve = {"bytes": 100}
        measured = {"unique_bytes": 90, "shared_bytes": 5, "host_bytes": 0}
        with patch.object(dc, "measure_resident_bytes", return_value=measured):
            result = rt.record_draft_model(target, SimpleNamespace())
        self.assertEqual(result["excess_bytes"], -10)
        self.assertEqual(coordinator.draft_resident, result)

    def test_settings_validate_budget_and_tolerance(self):
        base = {rt.PREFIX + "GIB": "32"}
        with patch.dict(
            os.environ, {**base, rt.PREFIX + "VRAM_BUDGET_GIB": "48"}, clear=True
        ):
            self.assertEqual(rt.Settings.from_env().vram_budget_gib, 48.0)
        for bad in (
            {rt.PREFIX + "VRAM_BUDGET_GIB": "-1"},
            {rt.PREFIX + "DRAFT_TOLERANCE": "2"},
        ):
            with (
                patch.dict(os.environ, {**base, **bad}, clear=True),
                self.assertRaises(ValueError),
            ):
                rt.Settings.from_env()


if __name__ == "__main__":
    unittest.main()
