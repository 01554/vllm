# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the global expert pool: step semantics, ownership."""

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
from lab_expert_tier import global_pool as gp  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


def make_sources(layers, experts, width=3):
    """Per-layer RAM banks: row e of layer l holds value l*100 + e in every tensor."""
    return [
        {
            name: torch.arange(experts, dtype=torch.int32)
            .add(layer * 100)
            .unsqueeze(1)
            .repeat(1, width)
            .contiguous()
            for name in gp.TENSORS
        }
        for layer in range(layers)
    ]


@unittest.skipIf(torch is None, "CPU torch is not installed")
class GlobalPoolTests(unittest.TestCase):
    def setup(self, layers=2, experts=6, slots=(2, 2), staging=2):
        device = torch.device("cpu")
        sources = make_sources(layers, experts)
        pool = gp.GlobalPool(device, sources[0], list(slots), staging)
        for layer, count in enumerate(slots):
            start = pool.offset(layer)
            for name in gp.TENSORS:
                pool.bank[name][start : start + count].copy_(
                    sources[layer][name][:count]
                )
        buffers = [
            gp.allocate_step_buffers(device, experts, staging) for _ in range(layers)
        ]
        return pool, sources, buffers

    def run_step(self, pool, sources, buffers, layer, ids):
        gp.step(pool.tables, layer, torch.tensor([ids]), buffers[layer])
        gp.copy_in(sources[layer], pool.bank, buffers[layer])
        return buffers[layer]

    def assert_bank_holds_owners(self, pool):
        tables = pool.tables
        E = tables.num_experts
        for row, key in enumerate(tables.row_key.tolist()):
            if key < 0:
                continue
            layer, expert = divmod(key, E)
            for name in gp.TENSORS:
                self.assertTrue(
                    bool((pool.bank[name][row] == layer * 100 + expert).all()),
                    f"{name} row {row} key {key}",
                )

    def test_initial_layout_packs_layers_and_validates(self):
        pool, sources, buffers = self.setup()
        tables = pool.tables
        self.assertEqual(
            tables.hot_phys.tolist(), [0, 1, -1, -1, -1, -1] + [2, 3, -1, -1, -1, -1]
        )
        self.assertEqual(tables.cold_phys.tolist(), [-1, -1, 2, 3, 4, 5] * 2)
        self.assertEqual(tables.row_key.tolist(), [0, 1, 6, 7, -1, -1])
        self.assertEqual(tables.staging_rows.tolist(), [4, 5])
        self.assertEqual(pool.snapshot(), [2, 2])
        self.assert_bank_holds_owners(pool)

    def test_gate_closed_stages_only_and_leaves_recency(self):
        pool, sources, buffers = self.setup()
        b = self.run_step(pool, sources, buffers, 1, [4, 4])
        self.assertEqual(int(b.promoted_count[0]), 0)
        self.assertEqual(int(b.staged_count[0]), 1)
        self.assertEqual(int(b.gather_count[0]), 1)
        self.assertEqual(b.step_map.tolist(), [2, 3, -1, -1, 4, -1])
        self.assertEqual(int(pool.tables.clock[0]), 0)
        self.assertEqual(pool.tables.last_use.sum().item(), 0)
        # Staging row 4 now holds layer 1 expert 4.
        self.assertTrue(bool((pool.bank[gp.TENSORS[0]][4] == 104).all()))
        self.assertEqual(pool.snapshot(), [2, 2])

    def test_miss_evicts_least_recent_of_any_layer_and_never_writes_ram(self):
        pool, sources, buffers = self.setup()
        before = [{n: t.clone() for n, t in s.items()} for s in sources]
        gp.set_gate(pool.tables, True)
        # Layer 0 touches its residents 0,1; layer 1 touches only 7 (key).
        self.run_step(pool, sources, buffers, 0, [0, 1])
        self.run_step(pool, sources, buffers, 1, [1, -1])
        # Layer 0 misses 5: victim is key 6 (layer 1 expert 0, last_use 0).
        b = self.run_step(pool, sources, buffers, 0, [5, 0])
        self.assertEqual(int(b.promoted_count[0]), 1)
        self.assertEqual(b.gather_src.tolist()[:1], [5])
        self.assertEqual(b.gather_dst.tolist()[:1], [2])
        tables = pool.tables
        self.assertEqual(tables.hot_phys.tolist()[5], 2)
        self.assertEqual(tables.hot_phys.tolist()[6], -1)
        self.assertEqual(tables.cold_phys.tolist()[6], 0)
        self.assertEqual(tables.row_key.tolist()[2], 5)
        self.assertEqual(b.step_map.tolist(), [0, 1, -1, -1, -1, 2])
        self.assertEqual(pool.snapshot(), [3, 1])
        self.assert_bank_holds_owners(pool)
        # Layer 1 now misses expert 0 again: victim is the least recent of
        # the remaining residents, layer 0's expert 1 (clock 1 < clock 3).
        b = self.run_step(pool, sources, buffers, 1, [0, 1])
        self.assertEqual(tables.hot_phys.tolist()[1], -1)
        self.assertEqual(tables.hot_phys.tolist()[6], 1)
        self.assertEqual(pool.snapshot(), [2, 2])
        self.assert_bank_holds_owners(pool)
        for layer, source in enumerate(sources):
            for name in gp.TENSORS:
                self.assertTrue(torch.equal(source[name], before[layer][name]))

    def test_no_victim_falls_back_to_staging(self):
        pool, sources, buffers = self.setup(layers=1, experts=4, slots=(2,), staging=3)
        gp.set_gate(pool.tables, True)
        # Every resident is selected, so the two misses have no victim.
        b = self.run_step(pool, sources, buffers, 0, [0, 1, 2, 3][:3])
        self.assertEqual(int(b.promoted_count[0]), 0)
        self.assertEqual(int(b.staged_count[0]), 1)
        self.assertEqual(b.step_map.tolist(), [0, 1, 2, -1])
        pool.snapshot()

    def test_invalid_ids_set_the_sticky_error(self):
        pool, sources, buffers = self.setup()
        self.run_step(pool, sources, buffers, 0, [9, 0])
        with self.assertRaises(RuntimeError):
            pool.snapshot()

    def test_host_swap_while_gated_matches_the_tables(self):
        pool, sources, buffers = self.setup()
        pool.host_swap(0, 0, 3)
        tables = pool.tables
        self.assertEqual(tables.hot_phys.tolist()[:6], [-1, 1, -1, 0, -1, -1])
        self.assertEqual(tables.cold_phys.tolist()[:6], [0, -1, 2, -1, 4, 5])
        self.assertEqual(int(tables.row_key[0]), 3)
        pool.snapshot()
        gp.set_gate(tables, True)
        with self.assertRaises(RuntimeError):
            pool.host_swap(0, 3, 0)

    def test_random_steps_keep_ownership_consistent(self):
        rng = random.Random(3)
        for trial in range(20):
            layers = rng.choice([1, 2, 3])
            experts = rng.choice([4, 6, 8])
            staging = rng.randint(1, 3)
            slots = tuple(rng.randint(1, experts - 1) for _ in range(layers))
            pool, sources, buffers = self.setup(layers, experts, slots, staging)
            gp.set_gate(pool.tables, rng.random() < 0.8)
            for step in range(30):
                layer = rng.randrange(layers)
                ids = [rng.choice([-1, rng.randrange(experts)]) for _ in range(staging)]
                b = self.run_step(pool, sources, buffers, layer, ids)
                resident = pool.snapshot()
                self.assertEqual(sum(resident), pool.tables.pool_rows)
                self.assert_bank_holds_owners(pool)
                for expert in {e for e in ids if e >= 0}:
                    self.assertGreaterEqual(int(b.step_map[expert]), 0)


if __name__ == "__main__":
    unittest.main()
