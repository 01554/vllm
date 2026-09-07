# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the bounded device-LRU planner."""

import importlib.util
import sys
import unittest
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import torch

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
from lab_expert_tier import device_lru as lru  # noqa: E402


class DeviceLRUTests(unittest.TestCase):
    def tables(self, experts=20, hot_slots=10, vram=8, ram=8):
        hot = torch.full((experts,), -1, dtype=torch.int32)
        cold = torch.full((experts,), -1, dtype=torch.int32)
        hot_experts = list(range(hot_slots))
        cold_experts = list(range(hot_slots, experts))
        hot[hot_experts] = torch.arange(hot_slots, dtype=torch.int32)
        cold[cold_experts] = torch.arange(len(cold_experts), dtype=torch.int32)
        return SimpleNamespace(
            hot_map=hot,
            cold_map=cold,
            last_use=torch.zeros(experts, dtype=torch.int64),
            clock=torch.zeros(1, dtype=torch.int64),
            error=torch.zeros(1, dtype=torch.int32),
            vram_free=torch.arange(vram, dtype=torch.int32),
            ram_free=torch.arange(ram, dtype=torch.int32),
        )

    def plan(self, ids, width=10, tables=None):
        tables = self.tables() if tables is None else tables
        state = lru.allocate_state(tables, width)
        lru.open_gate(state)
        return tables, lru.plan_step(
            torch.tensor(ids, dtype=torch.int64), tables, width
        )

    def test_ten_cold_misses_are_bounded_and_remaining_misses_are_staged(self):
        tables, plan = self.plan([list(range(10, 20))])
        self.assertEqual(int(plan.count[0]), 8)
        self.assertEqual(plan.promote_expert[:8].tolist(), list(range(10, 18)))
        self.assertEqual(plan.promote_cold_slot[:8].tolist(), list(range(10 - 10, 8)))
        self.assertEqual(int(plan.staged_only_count[0]), 2)
        self.assertEqual(plan.staged_only_expert[:2].tolist(), [18, 19])
        self.assertEqual(plan.staged_only_cold_slot[:2].tolist(), [8, 9])
        self.assertEqual(tables.hot_map.tolist(), list(range(10)) + [-1] * 10)

    def test_zero_capacity_startup_and_pending_only_stage(self):
        tables = self.tables(vram=0, ram=8)
        state = lru.allocate_state(tables, 10)
        lru.open_gate(state)
        plan = lru.plan_step(torch.arange(10, 20).reshape(1, 10), tables, 10)
        self.assertEqual(int(plan.count[0]), 0)
        self.assertEqual(plan.staged_only_expert[:10].tolist(), list(range(10, 20)))
        self.assertEqual(int(tables.clock[0]), 1)

        startup = self.tables()
        startup.last_use.copy_(torch.arange(20, dtype=torch.int64))
        startup.clock.fill_(7)
        state = lru.allocate_state(startup, 10)
        plan = lru.plan_step(torch.arange(10, 20).reshape(1, 10), startup, 10)
        self.assertEqual(int(plan.count[0]), 0)
        self.assertEqual(int(plan.staged_only_count[0]), 10)
        self.assertEqual(int(startup.clock[0]), 7)
        self.assertEqual(startup.last_use.tolist(), list(range(20)))

        pending = self.tables()
        state = lru.allocate_state(pending, 10)
        lru.open_gate(state)
        lru.set_pending(state, True)
        plan = lru.plan_step(torch.arange(10, 20).reshape(1, 10), pending, 10)
        self.assertEqual(int(plan.count[0]), 0)
        self.assertEqual(int(plan.staged_only_count[0]), 10)
        self.assertEqual(int(pending.clock[0]), 1)
        self.assertEqual(pending.last_use[10:].tolist(), [1] * 10)

    def test_all_hot_selection_protects_every_selected_hot_expert(self):
        tables = self.tables(experts=10, hot_slots=10, vram=8, ram=8)
        state = lru.allocate_state(tables, 10)
        lru.open_gate(state)
        plan = lru.plan_step(torch.tensor([[0, 1, 2, 2, 3]]), tables, 10)
        self.assertEqual(int(plan.count[0]), 0)
        self.assertEqual(int(plan.staged_only_count[0]), 0)
        self.assertEqual(tables.last_use[:4].tolist(), [1, 1, 1, 1])

    def test_victim_scan_includes_hot_slots_past_output_width(self):
        tables = self.tables(experts=40, hot_slots=30, vram=2, ram=2)
        state = lru.allocate_state(tables, 2)
        lru.open_gate(state)
        tables.last_use.copy_(torch.arange(40, dtype=torch.int64))
        tables.last_use[29] = -5
        plan = lru.plan_step(torch.tensor([[30, 31]]), tables, 2)
        self.assertEqual(int(plan.count[0]), 2)
        self.assertEqual(plan.victim_expert[:2].tolist(), [29, 0])
        self.assertEqual(plan.victim_hot_slot[:2].tolist(), [29, 0])

    def test_ties_are_broken_by_logical_hot_slot(self):
        tables = self.tables(experts=8, hot_slots=4, vram=2, ram=2)
        # The owners are intentionally permuted across logical slots.
        tables.hot_map.copy_(torch.tensor([3, 0, 2, 1, -1, -1, -1, -1]))
        tables.cold_map.copy_(torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3]))
        tables.last_use.fill_(4)
        state = lru.allocate_state(tables, 2)
        lru.open_gate(state)
        plan = lru.plan_step(torch.tensor([[4, 5]]), tables, 2)
        self.assertEqual(plan.victim_hot_slot[:2].tolist(), [0, 1])
        self.assertEqual(plan.victim_expert[:2].tolist(), [1, 3])

    def test_invalid_and_missing_owners_set_sticky_error_and_are_excluded(self):
        tables = self.tables(experts=6, hot_slots=3, vram=2, ram=2)
        tables.hot_map[4] = -1
        tables.cold_map[4] = -1
        state = lru.allocate_state(tables, 4)
        lru.open_gate(state)
        original = torch.tensor([[4, 99, -2, -1]], dtype=torch.int64)
        plan = lru.plan_step(original, tables, 4)
        self.assertEqual(int(tables.error[0]), 1)
        self.assertEqual(int(plan.count[0]), 0)
        self.assertEqual(int(plan.staged_only_count[0]), 0)
        self.assertTrue(torch.equal(original, torch.tensor([[4, 99, -2, -1]])))

    def test_duplicate_ids_update_recency_once_and_outputs_are_aliases(self):
        tables = self.tables()
        state = lru.allocate_state(tables, 4)
        self.assertIs(state.last_use, tables.last_use)
        self.assertIs(state.clock, tables.clock)
        self.assertIs(state.error, tables.error)
        lru.open_gate(state)
        ids = torch.tensor([[10, 10, 11, -1]], dtype=torch.int32)
        plan = lru.plan_step(ids, tables, 4)
        self.assertEqual(plan.promote_expert[:2].tolist(), [10, 11])
        self.assertEqual(tables.last_use[10:12].tolist(), [1, 1])
        self.assertEqual(ids.tolist(), [[10, 10, 11, -1]])
        with self.assertRaises(ValueError):
            lru.plan_step(torch.zeros((1, 5), dtype=torch.int32), tables, 4)


if __name__ == "__main__":
    unittest.main()
