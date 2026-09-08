# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused per-layer record must leave the device observer in the same
state as its own record_layer path, including errors and the finish."""

import importlib.util
import random
import sys
import unittest
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import cast

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
from lab_expert_tier import device_record as dr  # noqa: E402
from lab_expert_tier.heat_device import DeviceObserver  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

L, E, K, CAP = 3, 8, 2, 4


def make_observer():
    observer = DeviceObserver(num_layers=L, num_experts=E, decay=0.5, sync_period=1)
    observer.allocate(torch.device("cpu"), L, K, CAP, num_experts=E)
    return observer


def state(observer):
    acc = observer.accumulator
    fields = {
        "ids": observer.records,
        "activity": observer._activity_record,
        "valid": observer._valid_record,
        "counts": acc.counts,
        "route_total": acc._step_route_total,
        "route_hot": acc._step_route_hot,
        "valid_tokens": acc._step_valid_tokens,
        "expected": acc._expected_layer,
        "error": acc._error_flag,
        "seen": acc._step_hot_map_seen,
        "missing": acc._step_hot_map_missing,
        "heat": acc.heat,
        "cum_total": acc._route_total,
        "cum_hot": acc._route_hot,
    }
    return {name: tensor.clone() for name, tensor in fields.items()}


@unittest.skipIf(torch is None, "CPU torch is not installed")
class DeviceRecordTests(unittest.TestCase):
    def run_both(self, steps, finish_tokens=None):
        """steps: list of (layer, rows, ids, weights, padding, hot_map)."""
        classic, fused = make_observer(), make_observer()
        for layer, rows, ids, weights, padding, hot_map in steps:
            # The classic path is the observer plus the runtime's device
            # assertion (begin_layer); the fused record folds the latter into
            # the sticky error instead of raising.
            in_range = (ids >= 0) & (ids < E)
            allowed = in_range | ((ids == -1) & padding[:, None])
            good = (torch.isfinite(weights) & (weights >= 0)) | padding[:, None]
            classic.accumulator._error_flag.logical_or_(~(allowed & good).all())
            classic.record_layer(layer, ids, weights != 0, ~padding, rows, hot_map)
            dr.record(
                fused.kernel_targets(), layer, rows, ids, weights, padding, hot_map
            )
            fused.note_kernel_record(layer, rows, hot_map)
            self.assert_same(classic, fused)
        if finish_tokens is not None:
            classic.accumulator.finish_step(finish_tokens)
            fused.accumulator.finish_step(finish_tokens)
            self.assert_same(classic, fused)
        return classic, fused

    def assert_same(self, classic, fused):
        a, b = state(classic), state(fused)
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]), name)
        self.assertEqual(classic._recorded_layers, fused._recorded_layers)
        self.assertEqual(classic._hot_maps.keys(), fused._hot_maps.keys())

    @staticmethod
    def layer_inputs(rng, rows, invalid=False, bad_weight=False, padding=None):
        """One layer's routing; `padding` is shared by every layer of a forward."""
        ids = torch.tensor(
            [[rng.randrange(E) for _ in range(K)] for _ in range(rows)],
            dtype=torch.int32,
        )
        weights = torch.tensor(
            [[rng.choice([0.0, 0.25, 1.0]) for _ in range(K)] for _ in range(rows)]
        )
        if padding is None:
            padding = torch.tensor([r == rows - 1 and rows > 1 for r in range(rows)])
        padding = padding.clone()
        for r in range(rows):
            if padding[r]:
                ids[r] = -1
        if invalid:
            ids[0, 0] = E + 1
            padding[0] = False
        if bad_weight:
            weights[0, 1] = -1.0
            padding[0] = False
        hot_map = torch.tensor([rng.choice([-1, rng.randrange(20)]) for _ in range(E)])
        return ids, weights, padding, hot_map.to(torch.int32)

    def test_clean_forward_matches_including_finish(self):
        rng = random.Random(1)
        rows = 3
        steps = [(layer, rows, *self.layer_inputs(rng, rows)) for layer in range(L)]
        classic, fused = self.run_both(steps)
        self.assertFalse(bool(fused.accumulator._error_flag))
        self.assertGreater(int(fused.accumulator.counts.sum()), 0)
        classic.accumulator.finish_step(rows)
        fused.accumulator.finish_step(rows)
        self.assert_same(classic, fused)

    def test_duplicates_padding_and_missing_map_match(self):
        rng = random.Random(2)
        rows = 2
        ids = torch.tensor([[3, 3], [-1, -1]], dtype=torch.int32)
        weights = torch.tensor([[0.5, 0.5], [0.0, 0.0]])
        padding = torch.tensor([False, True])
        steps = [
            (0, rows, ids, weights, padding, None),
            (1, rows, *self.layer_inputs(rng, rows)),
            (2, rows, ids.clone(), weights.clone(), padding.clone(), None),
        ]
        classic, fused = self.run_both(steps)
        self.assertEqual(int(fused.accumulator.counts[0, 3]), 2)
        self.assertTrue(bool(fused.accumulator._step_hot_map_missing))
        self.assertFalse(bool(fused.accumulator._error_flag))
        fused.accumulator.finish_step(1)

    def test_invalid_id_bad_weight_and_layer_skip_set_the_error(self):
        rng = random.Random(3)
        for kind in ("invalid", "bad_weight", "inf", "neg_inf", "nan", "skip", "count"):
            rows = 2
            base = self.layer_inputs(rng, rows)
            if kind == "invalid":
                steps = [(0, rows, *self.layer_inputs(rng, rows, invalid=True))]
            elif kind == "bad_weight":
                steps = [(0, rows, *self.layer_inputs(rng, rows, bad_weight=True))]
            elif kind in ("inf", "neg_inf", "nan"):
                ids, weights, padding, hot_map = self.layer_inputs(rng, rows)
                weights[0, 0] = {"inf": float("inf"), "neg_inf": -float("inf")}.get(
                    kind, float("nan")
                )
                padding[0] = False
                steps = [(0, rows, ids, weights, padding, hot_map)]
            elif kind == "skip":
                steps = [(0, rows, *base), (2, rows, *self.layer_inputs(rng, rows))]
            else:
                second = self.layer_inputs(
                    rng, rows, padding=torch.tensor([True, True])
                )
                steps = [(0, rows, *base), (1, rows, *second)]
            with self.subTest(kind=kind):
                classic, fused = self.run_both(steps)
                self.assertTrue(bool(fused.accumulator._error_flag), kind)

    def test_non_finite_weights_on_padding_rows_are_tolerated(self):
        ids = torch.tensor([[-1, -1], [2, 3]], dtype=torch.int32)
        weights = torch.tensor([[float("inf"), float("nan")], [0.5, 0.5]])
        padding = torch.tensor([True, False])
        classic, fused = self.run_both([(0, 2, ids, weights, padding, None)])
        self.assertFalse(bool(fused.accumulator._error_flag))

    def test_second_forward_resets_at_layer_zero_like_the_observer(self):
        rng = random.Random(4)
        rows = 4
        steps = [(layer, rows, *self.layer_inputs(rng, rows)) for layer in range(L)]
        steps += [(layer, 1, *self.layer_inputs(rng, 1)) for layer in range(L)]
        classic, fused = self.run_both(steps, finish_tokens=1)
        self.assertFalse(bool(fused.accumulator._error_flag))


if __name__ == "__main__":
    unittest.main()
