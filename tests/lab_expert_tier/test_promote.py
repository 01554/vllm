# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for promote mode: planner reference, flip, ownership."""

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
from lab_expert_tier import promote as pm  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


def make_banks(hot_rows, ram_rows, width=3):
    bank = {
        name: torch.full((hot_rows, width), -1, dtype=torch.int32)
        for name in pm.TENSORS
    }
    ram = {
        name: torch.full((ram_rows, width), -1, dtype=torch.int32)
        for name in pm.TENSORS
    }
    return bank, ram


def fill_expert(banks, row, expert, i):
    for name in pm.TENSORS:
        banks[name][row].fill_(expert * 10 + i)


@unittest.skipIf(torch is None, "CPU torch is not installed")
class PromoteReferenceTests(unittest.TestCase):
    def setup(self, experts=6, hot=2, vram_free=2, ram_free=2, staging=2):
        device = torch.device("cpu")
        cold = experts - hot
        bank_rows = hot + staging + vram_free
        ram_rows = cold + ram_free
        tables = pm.allocate_tables(
            device,
            experts,
            hot,
            cold,
            range(hot + staging, bank_rows),
            range(cold, ram_rows),
        )
        bank, ram = make_banks(bank_rows, ram_rows)
        for e in range(hot):
            fill_expert(bank, e, e, 0)
        for e in range(hot, experts):
            fill_expert(ram, e - hot, e, 0)
        staging_rows = list(range(hot, hot + staging))
        return tables, bank, ram, staging_rows

    def run_step(self, tables, bank, ram, staging_rows, ids, enabled=True):
        plan = pm.reference_plan(ids, tables, len(staging_rows), enabled)
        gathers, staged, evicts, step_map = pm.apply_step_reference(
            tables, plan, staging_rows
        )
        pm.copy_rows_reference(ram, bank, gathers + staged)
        pm.copy_rows_reference(bank, ram, evicts)
        return plan, gathers, staged, evicts, step_map

    def assert_rows_hold_experts(self, tables, bank, ram):
        for expert in range(tables.hot_map.shape[0]):
            h = int(tables.hot_phys[expert])
            c = int(tables.cold_phys[expert])
            if h >= 0:
                self.assertEqual(int(bank[pm.TENSORS[0]][h][0]) // 10, expert)
            else:
                self.assertEqual(int(ram[pm.TENSORS[0]][c][0]) // 10, expert)

    def test_initial_tables_are_consistent_and_a_hit_step_changes_nothing(self):
        tables, bank, ram, staging_rows = self.setup()
        pm.check_tables(tables, 2, 4)
        plan, gathers, staged, evicts, step_map = self.run_step(
            tables, bank, ram, staging_rows, torch.tensor([[0, 1]])
        )
        self.assertEqual((int(plan.count[0]), int(plan.staged_only_count[0])), (0, 0))
        self.assertEqual((gathers, staged, evicts), ([], [], []))
        self.assertEqual(step_map.tolist(), [0, 1, -1, -1, -1, -1])
        self.assertEqual(int(tables.clock[0]), 1)
        self.assertEqual(tables.last_use.tolist()[:2], [1, 1])
        pm.check_tables(tables, 2, 4)

    def test_miss_promotes_least_recent_unselected_hot_and_keeps_shadow(self):
        tables, bank, ram, staging_rows = self.setup()
        # Touch expert 1 so expert 0 is the LRU victim.
        self.run_step(tables, bank, ram, staging_rows, torch.tensor([[1, 1]]))
        plan, gathers, staged, evicts, step_map = self.run_step(
            tables, bank, ram, staging_rows, torch.tensor([[4, 1]])
        )
        self.assertEqual(int(plan.count[0]), 1)
        self.assertEqual(int(plan.victim_expert[0]), 0)
        self.assertEqual(int(plan.promote_expert[0]), 4)
        # Expert 4 came from RAM row 2 into free VRAM row 4; expert 0 left
        # VRAM row 0 into the RAM pool head (row 4), and row 0 is now free.
        self.assertEqual(gathers, [(2, 4)])
        self.assertEqual(evicts, [(0, 4)])
        self.assertEqual(int(tables.hot_phys[4]), 4)
        self.assertEqual(int(tables.cold_phys[0]), 4)
        self.assertEqual(int(tables.hot_map[4]), 0)  # took expert 0's logical slot
        self.assertEqual(int(tables.cold_map[0]), 2)  # took expert 4's cold slot
        self.assertIn(0, tables.vram_free.tolist())
        # Expert 4's old RAM row 2 stays in the pool as its shadow.
        self.assertIn(2, tables.ram_free.tolist())
        self.assertEqual(int(tables.ram_shadow[2]), 4)
        self.assertEqual(step_map.tolist()[4], 4)
        pm.check_tables(tables, 2, 4)
        self.assert_rows_hold_experts(tables, bank, ram)
        # Evicting expert 4 again finds its shadow intact: no copy.
        plan, gathers, staged, evicts, step_map = self.run_step(
            tables, bank, ram, staging_rows, torch.tensor([[0, 1]])
        )
        self.assertEqual(int(plan.victim_expert[0]), 4)
        self.assertEqual(evicts, [])
        self.assertEqual(int(tables.cold_phys[4]), 2)
        pm.check_tables(tables, 2, 4)
        self.assert_rows_hold_experts(tables, bank, ram)

    def test_overflow_misses_are_staged_only_and_gate_blocks_promotion(self):
        tables, bank, ram, staging_rows = self.setup(hot=1, vram_free=1, staging=2)
        # Two misses, one free VRAM row: one promotes, the other stages.
        plan, gathers, staged, evicts, step_map = self.run_step(
            tables, bank, ram, staging_rows, torch.tensor([[3, 5]])
        )
        self.assertEqual((int(plan.count[0]), int(plan.staged_only_count[0])), (1, 1))
        self.assertEqual(int(plan.staged_only_expert[0]), 5)
        self.assertEqual(staged, [(4, 1)])  # expert 5's RAM row into row 1
        self.assertEqual(step_map.tolist()[5], 1)  # first staging row
        self.assertEqual(int(bank[pm.TENSORS[0]][1][0]) // 10, 5)
        pm.check_tables(tables, 1, 5)
        # Gate closed: everything stays staged, tables untouched but clock.
        before = tables.hot_map.clone()
        plan, gathers, staged, evicts, step_map = self.run_step(
            tables, bank, ram, staging_rows, torch.tensor([[4, 5]]), enabled=False
        )
        self.assertEqual(int(plan.count[0]), 0)
        self.assertEqual(int(plan.staged_only_count[0]), 2)
        self.assertTrue(torch.equal(tables.hot_map, before))
        self.assertEqual(int(tables.clock[0]), 1)  # startup: recency untouched
        pm.check_tables(tables, 1, 5)

    def test_flip_rejects_a_plan_beyond_the_pool_capacity(self):
        tables, bank, ram, staging_rows = self.setup(hot=3, vram_free=3, ram_free=1)
        plan = pm.reference_plan(torch.tensor([[3, 4]]), tables, 2)
        oversized = pm.StepPlan(
            promote_expert=torch.tensor([3, 4]),
            promote_cold_slot=torch.tensor([0, 1]),
            victim_expert=torch.tensor([0, 1]),
            victim_hot_slot=torch.tensor([0, 1]),
            count=torch.tensor([2], dtype=torch.int32),
            staged_only_expert=plan.staged_only_expert,
            staged_only_cold_slot=plan.staged_only_cold_slot,
            staged_only_count=torch.tensor([0], dtype=torch.int32),
        )
        with self.assertRaises(RuntimeError):
            pm.apply_step_reference(tables, oversized, staging_rows)

    def test_invalid_and_duplicate_ids_never_plan_or_index(self):
        tables, bank, ram, staging_rows = self.setup()
        plan, gathers, staged, evicts, step_map = self.run_step(
            tables, bank, ram, staging_rows, torch.tensor([[-1, 99]])
        )
        self.assertEqual((int(plan.count[0]), int(plan.staged_only_count[0])), (0, 0))
        plan, gathers, staged, evicts, step_map = self.run_step(
            tables, bank, ram, staging_rows, torch.tensor([[3, 3]])
        )
        self.assertEqual(int(plan.count[0]), 1)
        self.assertEqual(int(plan.promote_expert[0]), 3)
        pm.check_tables(tables, 2, 4)

    def test_random_steps_keep_ownership_and_bytes_consistent(self):
        rng = random.Random(5)
        for trial in range(30):
            experts = rng.choice([4, 6, 8])
            hot = rng.randint(1, experts - 1)
            vram_free = rng.randint(1, 3)
            staging = rng.randint(1, 3)
            ram_free = rng.randint(1, 3)
            tables, bank, ram, staging_rows = self.setup(
                experts, hot, vram_free, ram_free, staging
            )
            for step in range(25):
                k = staging  # rows*k must fit the staging rows
                ids = torch.tensor(
                    [[rng.choice([-1, rng.randrange(experts)]) for _ in range(k)]]
                )
                self.run_step(tables, bank, ram, staging_rows, ids)
                pm.check_tables(tables, hot, experts - hot)
                self.assert_rows_hold_experts(tables, bank, ram)
                # Every valid selected expert is hot or staged this step.


if __name__ == "__main__":
    unittest.main()


def smoke_sequence():
    """Fixed batch-1 routing steps for a 24-expert layer, 12 hot, top-k 10,
    8 free VRAM rows, 8 pool rows. Reused by the GPU smoke: step 1 promotes 8
    and stages 2; step 2 promotes 8 more whose victims include experts
    promoted in step 1, so most evictions reclaim intact shadows without a
    copy; steps 3-5 churn the pool so shadows get invalidated and later
    re-misses copy again."""
    import torch

    return [
        torch.tensor([[12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]),
        torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        torch.tensor([[20, 21, 22, 23, 10, 11, 18, 19, 8, 9]]),
        torch.tensor([[12, 13, 14, 15, 16, 17, 0, 1, 2, 3]]),
        torch.tensor([[4, 5, 6, 7, 20, 21, 22, 23, 10, 11]]),
    ]


@unittest.skipIf(torch is None, "CPU torch is not installed")
class PromoteSmokeSequenceTests(PromoteReferenceTests):
    def test_smoke_sequence_reference_covers_promote_stage_reclaim_and_churn(self):
        tables, bank, ram, staging_rows = self.setup(
            experts=24, hot=12, vram_free=8, ram_free=8, staging=10
        )
        expected_counts = []
        for ids in smoke_sequence():
            plan, gathers, staged, evicts, step_map = self.run_step(
                tables, bank, ram, staging_rows, ids
            )
            expected_counts.append(
                (int(plan.count[0]), int(plan.staged_only_count[0]), len(evicts))
            )
            pm.check_tables(tables, 12, 12)
            self.assert_rows_hold_experts(tables, bank, ram)
            # Every selected expert is served: hot after the flip or staged.
            for expert in {int(e) for e in ids.reshape(-1).tolist()}:
                self.assertGreaterEqual(int(step_map[expert]), 0)
        # Step 1: 8 promoted, 2 staged, 8 victims written. Step 2: 8 promoted
        # again; six of the victims are step-1 promotions whose shadows are
        # intact, so only two D2H copies. Churn afterwards invalidates
        # shadows, so a later step copies again.
        self.assertEqual(expected_counts[0], (8, 2, 8))
        self.assertEqual(expected_counts[1][:2], (8, 0))
        self.assertLess(expected_counts[1][2], 8)
        self.assertTrue(any(evicts > 0 for _, _, evicts in expected_counts[2:]))
