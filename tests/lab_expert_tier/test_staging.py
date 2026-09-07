# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for decode-time cold expert staging."""

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
from lab_expert_tier import staging as st  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


def reference_plan(ids, cold_map, hot_map, hot_slots):
    """Sequential semantics: distinct cold experts in first-occurrence order."""
    experts: list[int] = []
    for row in ids:
        for e in row:
            expert = int(e)
            if (
                0 <= expert < len(hot_map)
                and int(cold_map[expert]) >= 0
                and expert not in experts
            ):
                experts.append(expert)
    expert_map = [int(v) for v in hot_map]
    gather = []
    for slot, expert in enumerate(experts):
        expert_map[expert] = hot_slots + slot
        gather.append(int(cold_map[expert]))
    return gather, expert_map


@unittest.skipIf(torch is None, "CPU torch is not installed")
class StagingPlanTests(unittest.TestCase):
    def maps(self, num_experts=8, hot_slots=3):
        hot = [i if i < hot_slots else -1 for i in range(num_experts)]
        cold = [-1 if i < hot_slots else i - hot_slots for i in range(num_experts)]
        return (
            torch.tensor(hot, dtype=torch.int32),
            torch.tensor(cold, dtype=torch.int32),
        )

    def test_plan_collapses_duplicates_ignores_invalid_and_rebuilds_map(self):
        hot_map, cold_map = self.maps()
        # Hot 1 and 2, cold 5 twice, cold 7, one padding sentinel.
        ids = torch.tensor([[5, 1, 7, 5, 2, -1]], dtype=torch.int32)
        gather, expert_map, count = st.plan_staging(ids, cold_map, hot_map, 3, 6)
        self.assertEqual(int(count), 2)
        self.assertEqual(gather.shape, (6,))
        self.assertEqual(expert_map.shape, (8,))
        self.assertEqual(gather[:2].tolist(), [2, 4])  # cold slots of 5 and 7
        self.assertEqual(gather[2:].tolist(), [0, 0, 0, 0])
        self.assertEqual(expert_map.tolist(), [0, 1, 2, -1, -1, 3, -1, 4])
        self.assertEqual(count.shape, (1,))
        # The input maps are untouched: the plan never carries over.
        self.assertEqual(hot_map.tolist(), [0, 1, 2, -1, -1, -1, -1, -1])
        self.assertEqual(expert_map.dtype, torch.int32)
        self.assertEqual(gather.dtype, torch.int64)

    def test_plan_zero_miss_all_padding_and_full_capacity(self):
        hot_map, cold_map = self.maps()
        gather, expert_map, count = st.plan_staging(
            torch.tensor([[0, 1, 2, 1]]), cold_map, hot_map, 3, 4
        )
        self.assertEqual(int(count), 0)
        self.assertEqual(expert_map.tolist(), hot_map.tolist())
        gather, expert_map, count = st.plan_staging(
            torch.tensor([[-1, -1, -1, -1]]), cold_map, hot_map, 3, 4
        )
        self.assertEqual(int(count), 0)
        self.assertEqual(expert_map.tolist(), hot_map.tolist())
        # Every selection cold and distinct: all staging rows are used.
        gather, expert_map, count = st.plan_staging(
            torch.tensor([[7, 3, 6, 4]]), cold_map, hot_map, 3, 4
        )
        self.assertEqual(int(count), 4)
        # First-occurrence order: 7, 3, 6, 4 take staging slots 0..3.
        self.assertEqual(gather.tolist(), [4, 0, 3, 1])
        self.assertEqual(expert_map.tolist(), [0, 1, 2, 4, 6, -1, 5, 3])
        with self.assertRaises(ValueError):
            st.plan_staging(torch.tensor([[7, 3, 6, 4, 5]]), cold_map, hot_map, 3, 4)
        with self.assertRaises(ValueError):
            st.plan_staging(torch.zeros(1, st.PLAN_WIDTH + 1), cold_map, hot_map, 3, 32)

    def test_plan_out_of_range_ids_never_index_and_never_stage(self):
        hot_map, cold_map = self.maps()
        ids = torch.tensor([[99, -7, 6, 8]], dtype=torch.int64)
        gather, expert_map, count = st.plan_staging(ids, cold_map, hot_map, 3, 4)
        self.assertEqual(int(count), 1)
        self.assertEqual(gather[0].item(), 3)
        self.assertEqual(expert_map.tolist(), [0, 1, 2, -1, -1, -1, 3, -1])

    def test_plan_matches_sequential_reference_on_random_routes(self):
        rng = random.Random(3)
        for _ in range(300):
            experts = rng.choice([4, 8, 16, 64])
            hot_slots = rng.randint(1, experts - 1)
            order = list(range(experts))
            rng.shuffle(order)
            hot = [-1] * experts
            cold = [-1] * experts
            for slot, expert in enumerate(order[:hot_slots]):
                hot[expert] = slot
            for slot, expert in enumerate(order[hot_slots:]):
                cold[expert] = slot
            hot_map = torch.tensor(hot, dtype=torch.int32)
            cold_map = torch.tensor(cold, dtype=torch.int32)
            k = rng.randint(1, 8)
            rows = rng.choice([1, 1, 1, 2])
            capacity = rows * k
            ids = [
                [
                    rng.choice([-1, rng.randrange(experts), rng.randrange(experts)])
                    for _ in range(k)
                ]
                for _ in range(rows)
            ]
            gather, expert_map, count = st.plan_staging(
                torch.tensor(ids), cold_map, hot_map, hot_slots, capacity
            )
            ref_gather, ref_map = reference_plan(ids, cold, hot, hot_slots)
            self.assertEqual(int(count), len(ref_gather))
            self.assertEqual(gather[: len(ref_gather)].tolist(), ref_gather)
            self.assertEqual(expert_map.tolist(), ref_map)
            # Staged slots are dense, in bounds, and unique.
            staged = [v for v in expert_map.tolist() if v >= hot_slots]
            self.assertEqual(
                sorted(staged), list(range(hot_slots, hot_slots + len(staged)))
            )


@unittest.skipIf(torch is None, "CPU torch is not installed")
class StagingGatherTests(unittest.TestCase):
    def test_reference_gather_copies_exactly_count_rows_of_every_tensor(self):
        cold_rows, staging_rows, width = 5, 3, 4
        source = {
            name: (torch.arange(cold_rows * width) + 100 * i).reshape(cold_rows, width)
            for i, name in enumerate(st.TENSORS)
        }
        staging = {name: torch.full((staging_rows, width), -1) for name in st.TENSORS}
        gather = torch.tensor([4, 1, 0])
        st.gather_staging(source, staging, gather, torch.tensor(2, dtype=torch.int32))
        for name in st.TENSORS:
            self.assertTrue(torch.equal(staging[name][0], source[name][4]))
            self.assertTrue(torch.equal(staging[name][1], source[name][1]))
            # The third slot was not planned: it keeps whatever it held.
            self.assertTrue(torch.equal(staging[name][2], torch.full((width,), -1)))

    def test_byte_rows_view_requires_contiguous_rows(self):
        tensor = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        rows = st._byte_rows(tensor)
        self.assertEqual(rows.shape, (3, 16))
        self.assertEqual(rows.dtype, torch.uint8)
        with self.assertRaises(ValueError):
            st._byte_rows(tensor.t())


if __name__ == "__main__":
    unittest.main()
