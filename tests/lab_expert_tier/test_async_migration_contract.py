# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for asynchronous expert migration primitives."""

import importlib.util
import sys
import threading
import unittest
from collections import Counter
from dataclasses import dataclass
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

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

from lab_expert_tier import runtime as rt  # noqa: E402
from lab_expert_tier.async_migration import (  # noqa: E402
    MigrationTransaction,
    SpareRing,
    SpareRow,
    _NullStream,
    preflight,
)
from lab_expert_tier.tier_policy import Swap, SwapPlan  # noqa: E402


class _RecordingNullStream(_NullStream):
    def __init__(self):
        self.waited: list[Any] = []

    def wait_event(self, event):
        self.waited.append(event)


class _GateEvent:
    def __init__(self, done=False):
        self.done = done
        self.synchronized = 0

    def query(self):
        return self.done

    def synchronize(self):
        self.synchronized += 1
        self.done = True


@dataclass(frozen=True)
class _Swap:
    layer: int
    hot_slot: int
    cold_slot: int


def _cpu_layer():
    """Build a real TierLayer shell around small CPU row tensors."""
    layer = object.__new__(rt.TierLayer)
    layer.device = torch.device("cpu")
    layer.num_experts = 3
    layer.hot_slots = 1
    layer.cold_slots = 2
    layer.hot_rows = [0]
    layer.cold_rows = [0, 1]
    layer.hot_map_host = (0, -1, -1)
    layer.cold_map_host = (-1, 0, 1)
    layer.hot_map = None
    layer.cold_map = None
    layer.row_bytes = 1
    layer.bank = {}
    layer.cold_cpu = {}
    layer.cold = {}
    for offset, name in enumerate(rt.TENSORS):
        bank = torch.arange(6, dtype=torch.int64).reshape(3, 2) + offset * 100
        cold = torch.arange(6, dtype=torch.int64).reshape(3, 2) + offset * 1000
        layer.bank[name] = bank
        layer.cold_cpu[name] = cold.clone()
        layer.cold[name] = layer.cold_cpu[name]
    layer.vram_spares = SpareRing([2])
    layer.ram_spares = SpareRing([2])
    return layer


def _cpu_coordinator(layer, events):
    """Build a TierCoordinator shell so its real transaction methods run."""
    coordinator = object.__new__(rt.TierCoordinator)
    coordinator.layers = [layer]
    coordinator.device = torch.device("cpu")
    coordinator.lock = threading.Lock()
    coordinator.pending = None
    coordinator.stream = None
    coordinator.stream_id = None
    coordinator.recorded = 0
    coordinator.forward_rows = None
    coordinator.poisoned = False
    coordinator.per_layer_swaps = [0]
    coordinator.stats = Counter()
    coordinator.heat_enabled = False

    class Policy:
        tokens_total = 0
        version = 0
        last_sync_tokens = 0

        def commit(self, plan):
            events.append("commit")
            self.expert_to_hot = [list(layer.hot_map_host)]
            self.expert_to_cold = [list(layer.cold_map_host)]

    class Observer:
        capacity = 1

        def finish(
            self, rows, valid_rows, heat_enabled, stream, num_experts, is_decode=None
        ):
            events.append("finish")
            return rt.Deferred()

        def flush(self):
            events.append("flush")
            return rt.Deferred()

        def rebase(self, **metadata):
            events.append("rebase")

    policy = Policy()
    policy.expert_to_hot = [list(layer.hot_map_host)]
    policy.expert_to_cold = [list(layer.cold_map_host)]
    coordinator.policy = policy
    coordinator.observer = Observer()
    coordinator.settings = SimpleNamespace(stats_every=100)
    return coordinator


class AsyncMigrationPrimitiveTests(unittest.TestCase):
    def test_empty_plan_is_ineligible(self):
        result = preflight([], {})

        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, "empty")
        self.assertEqual(result.per_layer, {})

    def test_over_budget_rejects_the_whole_plan_without_truncation(self):
        swaps = [
            _Swap(layer=0, hot_slot=0, cold_slot=3),
            _Swap(layer=0, hot_slot=1, cold_slot=4),
            _Swap(layer=1, hot_slot=0, cold_slot=2),
        ]

        result = preflight(swaps, {0: 1, 1: 1})

        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, "over_budget")
        self.assertEqual(result.per_layer[0], tuple(swaps[:2]))
        self.assertEqual(result.per_layer[1], (swaps[2],))

    def test_hot_or_cold_slot_reuse_rejects_the_whole_layer_plan(self):
        cases = (
            (
                "hot",
                [
                    _Swap(layer=0, hot_slot=2, cold_slot=5),
                    _Swap(layer=0, hot_slot=2, cold_slot=6),
                ],
            ),
            (
                "cold",
                [
                    _Swap(layer=0, hot_slot=2, cold_slot=5),
                    _Swap(layer=0, hot_slot=3, cold_slot=5),
                ],
            ),
        )
        for name, swaps in cases:
            with self.subTest(name=name):
                result = preflight(swaps, {0: 2})

                self.assertFalse(result.eligible)
                self.assertEqual(result.reason, "slot_reuse")
                self.assertEqual(result.per_layer[0], tuple(swaps))

    def test_order_and_layer_partition_are_preserved(self):
        swaps = [
            _Swap(layer=1, hot_slot=7, cold_slot=1),
            _Swap(layer=0, hot_slot=4, cold_slot=0),
            _Swap(layer=1, hot_slot=8, cold_slot=2),
            _Swap(layer=0, hot_slot=5, cold_slot=3),
        ]

        result = preflight(swaps, {0: 2, 1: 2})

        self.assertTrue(result.eligible)
        self.assertIsNone(result.reason)
        self.assertEqual(list(result.per_layer), [1, 0])
        self.assertEqual(result.per_layer[1], (swaps[0], swaps[2]))
        self.assertEqual(result.per_layer[0], (swaps[1], swaps[3]))

    def test_same_slots_in_different_layers_are_independent(self):
        swaps = [
            _Swap(layer=0, hot_slot=0, cold_slot=1),
            _Swap(layer=1, hot_slot=0, cold_slot=1),
        ]

        result = preflight(swaps, {0: 1, 1: 1})

        self.assertTrue(result.eligible)
        self.assertEqual(result.per_layer[0], (swaps[0],))
        self.assertEqual(result.per_layer[1], (swaps[1],))

    def test_destination_row_is_not_free_until_explicit_return(self):
        ring = SpareRing([10, 11])

        destination = ring.pop()
        self.assertEqual(destination, SpareRow(row=10))
        self.assertEqual(ring.free, 1)

        # Transfer completion alone has no ring side effect.  The retired row
        # is returned only by the explicit push carrying its compute fence.
        self.assertEqual(ring.pop(), SpareRow(row=11))
        self.assertEqual(ring.free, 0)
        compute_fence = object()
        ring.push(destination.row, compute_fence)
        self.assertEqual(ring.free, 1)
        returned = ring.pop()
        self.assertEqual(returned.row, 10)
        self.assertIs(returned.fence, compute_fence)

    def test_spare_ring_is_fifo_and_empty_pop_fails(self):
        ring = SpareRing([4])

        self.assertEqual(ring.pop(), SpareRow(row=4))
        with self.assertRaises(RuntimeError):
            ring.pop()

    def test_migration_transaction_starts_enqueued_with_owned_entries(self):
        plan = (_Swap(layer=0, hot_slot=0, cold_slot=1),)
        result = preflight(plan, {0: 1})
        transaction = MigrationTransaction(plan=plan, preflight=result)

        self.assertEqual(transaction.plan, plan)
        self.assertIs(transaction.preflight, result)
        self.assertEqual(transaction.entries, [])
        self.assertEqual(transaction.state, "enqueued")
        transaction.entries.append((0, plan[0], SpareRow(2), SpareRow(8)))
        another = MigrationTransaction(plan=plan, preflight=result)
        self.assertEqual(another.entries, [])


class AsyncMigrationRuntimeTests(unittest.TestCase):
    def test_enqueue_keeps_old_maps_and_uses_unpublished_spare_rows(self):
        layer = _cpu_layer()
        old_hot_map = layer.hot_map_host
        old_cold_map = layer.cold_map_host
        old_hot_rows = list(layer.hot_rows)
        old_cold_rows = list(layer.cold_rows)
        hot_before = {name: tensor[0].clone() for name, tensor in layer.bank.items()}
        cold_before = {
            name: tensor[0].clone() for name, tensor in layer.cold_cpu.items()
        }
        swap = Swap(0, 0, 0, old_expert=0, new_expert=1)
        stream = _RecordingNullStream()

        vram_spare, ram_spare = layer.enqueue_swap(swap, stream)

        self.assertEqual(vram_spare, SpareRow(2))
        self.assertEqual(ram_spare, SpareRow(2))
        self.assertEqual(layer.hot_map_host, old_hot_map)
        self.assertEqual(layer.cold_map_host, old_cold_map)
        self.assertEqual(layer.hot_rows, old_hot_rows)
        self.assertEqual(layer.cold_rows, old_cold_rows)
        self.assertEqual(layer.vram_spares.free, 0)
        self.assertEqual(layer.ram_spares.free, 0)
        for name in rt.TENSORS:
            self.assertTrue(torch.equal(layer.bank[name][2], cold_before[name]))
            self.assertTrue(torch.equal(layer.cold_cpu[name][2], hot_before[name]))

    def test_flip_retires_rows_with_fence_and_waits_before_reuse(self):
        layer = _cpu_layer()
        stream = _RecordingNullStream()
        first = Swap(0, 0, 0, old_expert=0, new_expert=1)
        vram_spare, ram_spare = layer.enqueue_swap(first, stream)
        retire_fence = object()

        layer.flip_swap(first, vram_spare, ram_spare, retire_fence)

        self.assertEqual(layer.hot_rows, [2])
        self.assertEqual(layer.cold_rows, [2, 1])
        self.assertEqual(layer.hot_map_host, (-1, 0, -1))
        self.assertEqual(layer.cold_map_host, (0, -1, 1))
        self.assertEqual(layer.vram_spares.free, 1)
        self.assertEqual(layer.ram_spares.free, 1)

        second = Swap(0, 0, 0, old_expert=1, new_expert=0)
        second_vram, second_ram = layer.enqueue_swap(second, stream)

        self.assertEqual(second_vram.row, 0)
        self.assertEqual(second_ram.row, 0)
        self.assertEqual(stream.waited, [retire_fence, retire_fence])

    def test_pending_transfer_keeps_old_maps_until_settled(self):
        events: list[str] = []
        layer = _cpu_layer()
        coordinator = _cpu_coordinator(layer, events)
        swap = Swap(0, 0, 0, old_expert=0, new_expert=1)
        plan = SwapPlan(swaps=(swap,), tokens_total=0, base_version=0)
        verdict = preflight(plan.swaps, {0: 1})
        coordinator._begin_async(plan, verdict)
        gate = _GateEvent(done=False)
        coordinator.pending.transfer_event = gate
        old_maps = (layer.hot_map_host, layer.cold_map_host)

        self.assertFalse(coordinator.settle_pending(wait=False))
        self.assertEqual((layer.hot_map_host, layer.cold_map_host), old_maps)
        self.assertIsNotNone(coordinator.pending)
        self.assertEqual(gate.synchronized, 0)
        self.assertEqual(events, [])

        gate.done = True
        self.assertTrue(coordinator.settle_pending(wait=False))
        self.assertIsNone(coordinator.pending)
        self.assertEqual(events, ["commit", "rebase"])

    def test_enqueue_failure_leaves_partial_transaction_visible(self):
        events: list[str] = []
        first_layer = _cpu_layer()
        coordinator = _cpu_coordinator(first_layer, events)

        class FailingLayer:
            def enqueue_swap(self, swap, stream):
                raise RuntimeError("copy failed")

        coordinator.layers = [first_layer, FailingLayer()]
        coordinator.per_layer_swaps = [0, 0]
        swaps = (
            Swap(0, 0, 0, old_expert=0, new_expert=1),
            Swap(1, 0, 0, old_expert=0, new_expert=1),
        )
        plan = SwapPlan(swaps=swaps, tokens_total=0, base_version=0)
        verdict = preflight(plan.swaps, {0: 1, 1: 1})

        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            coordinator._begin_async(plan, verdict)

        self.assertIsNotNone(coordinator.pending)
        self.assertEqual(len(coordinator.pending.entries), 1)

    def test_coordinator_commits_and_rebases_before_observer_finish(self):
        events: list[str] = []
        layer = _cpu_layer()
        coordinator = _cpu_coordinator(layer, events)
        swap = Swap(0, 0, 0, old_expert=0, new_expert=1)
        plan = SwapPlan(swaps=(swap,), tokens_total=0, base_version=0)
        verdict = preflight(plan.swaps, {0: 1})
        coordinator._begin_async(plan, verdict)

        coordinator._finish_forward(rows=1, valid_rows=1)

        self.assertEqual(events, ["commit", "rebase", "finish"])
        self.assertIsNone(coordinator.pending)

    def test_flush_settles_pending_before_observer_flush(self):
        events: list[str] = []
        layer = _cpu_layer()
        coordinator = _cpu_coordinator(layer, events)
        swap = Swap(0, 0, 0, old_expert=0, new_expert=1)
        plan = SwapPlan(swaps=(swap,), tokens_total=0, base_version=0)
        verdict = preflight(plan.swaps, {0: 1})
        coordinator._begin_async(plan, verdict)

        coordinator.flush(plan=False)

        self.assertEqual(events, ["commit", "rebase", "flush"])
        self.assertIsNone(coordinator.pending)


if __name__ == "__main__":
    unittest.main()
