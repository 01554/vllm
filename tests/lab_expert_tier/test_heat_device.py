# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU differential tests for device heat accumulation and snapshot import."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "vllm" / "_lab_expert_tier")
)
from heat_device import Deferred, DeviceHeatAccumulator, DeviceObserver, DeviceSnapshot
from tier_policy import TierPolicy


def _tensors(routes, activity, valid):
    ids = torch.tensor(routes, dtype=torch.int64)
    flags = torch.tensor(activity, dtype=torch.bool)
    mask = torch.tensor(valid, dtype=torch.bool)
    return ids, flags, mask


def _record(accumulator, routes, activity, valid, num_tokens):
    ids, flags, mask = _tensors(routes[0], activity[0], valid)
    for layer, (layer_routes, layer_activity) in enumerate(zip(routes, activity)):
        ids, flags, mask = _tensors(layer_routes, layer_activity, valid)
        accumulator.record_layer(layer, ids, flags, mask, len(valid))
    accumulator.finish_step(num_tokens)


class DeviceHeatTests(unittest.TestCase):
    def test_integer_duplicate_masked_and_zero_activity_counts_match_numpy(self):
        routes = [
            [[0, 0, 3], [1, 2, 2], [-1, -1, -1], [2, 3, 1]],
            [[3, 1, 1], [2, 2, 0], [-1, -1, -1], [0, 3, 3]],
        ]
        activity = [
            [[1, 1, 0], [1, 0, 1], [0, 0, 0], [0, 1, 1]],
            [[1, 0, 1], [1, 1, 1], [0, 0, 0], [1, 0, 1]],
        ]
        valid = [True, True, False, False]
        weights = [
            [[1.0, 2.0, 0.0], [1.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.0, 2.0, 1.0]],
            [[1.0, 0.0, 2.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
        ]
        policy = TierPolicy(
            2,
            4,
            2,
            decay=0.5,
            sync_period=50,
            initial_scores=[[2.0, 1.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]],
        )
        accumulator = DeviceHeatAccumulator(
            2,
            4,
            decay=0.5,
            sync_period=50,
            initial_scores=[[2.0, 1.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]],
            top_k=3,
            max_rows=4,
            session_id="differential",
        )

        _record(accumulator, routes, activity, valid, 2)
        policy.observe_step(routes, 2, weights, valid)
        self.assertEqual(accumulator.error, False)
        self.assertEqual(
            tuple(
                tuple(value.hex() for value in row) for row in accumulator.heat.tolist()
            ),
            tuple(tuple(value.hex() for value in row) for row in policy.heat),
        )
        self.assertEqual(accumulator.tokens_total, policy.tokens_total)
        self.assertEqual(accumulator.last_step_tokens, 2)

    def test_zero_token_dummy_does_not_decay_or_earn_credit(self):
        initial = [[7.0, 1.0, 0.0]]
        policy = TierPolicy(1, 3, 1, decay=0.25, sync_period=1, initial_scores=initial)
        accumulator = DeviceHeatAccumulator(
            1,
            3,
            decay=0.25,
            sync_period=1,
            initial_scores=initial,
            top_k=2,
            max_rows=2,
            session_id="dummy",
        )
        routes = [[[-1, -1], [-1, -1]]]
        activity = [[[False, False], [False, False]]]
        valid = [False, False]
        _record(accumulator, routes, activity, valid, 0)
        policy.observe_step(routes, 0, [[[0.0, 0.0], [0.0, 0.0]]], valid)
        self.assertEqual(accumulator.tokens_total, 0)
        self.assertEqual(accumulator.last_step_tokens, 0)
        self.assertEqual(accumulator.heat.tolist(), policy._heat.tolist())
        self.assertFalse(accumulator.resync_due)

    def test_disabled_positive_dummy_count_does_not_poison_first_enabled_step(self):
        accumulator = DeviceHeatAccumulator(
            1,
            2,
            sync_period=1,
            top_k=1,
            max_rows=2,
            enabled=False,
            session_id="disabled-positive-dummy",
        )
        dummy_ids = torch.tensor([[-1], [-1]], dtype=torch.int64)
        dummy_active = torch.zeros((2, 1), dtype=torch.bool)
        dummy_valid = torch.zeros((2,), dtype=torch.bool)
        accumulator.record_layer(0, dummy_ids, dummy_active, dummy_valid, 2)
        # The host knows two padded rows, while the all-false device mask has
        # zero valid tokens.  Startup remains unverified only if it records a
        # structural or routing error; this metadata mismatch is expected.
        accumulator.finish_step(2)
        self.assertFalse(accumulator.error)

        accumulator.on_heat_enabled()
        ids = torch.tensor([[0]], dtype=torch.int64)
        active = torch.ones((1, 1), dtype=torch.bool)
        valid = torch.ones((1,), dtype=torch.bool)
        accumulator.record_layer(0, ids, active, valid, 1)
        accumulator.finish_step(1)
        snapshot = accumulator.flush()
        self.assertIsNotNone(snapshot)
        self.assertTrue(snapshot.verified)
        self.assertFalse(snapshot.error)
        self.assertEqual(snapshot.tokens, 1)

    def test_disabled_structural_and_id_errors_remain_sticky_after_enable(self):
        accumulator = DeviceHeatAccumulator(
            1,
            2,
            sync_period=1,
            top_k=1,
            max_rows=1,
            enabled=False,
            session_id="disabled-errors",
        )
        invalid_ids = torch.tensor([[99]], dtype=torch.int64)
        active = torch.ones((1, 1), dtype=torch.bool)
        valid = torch.ones((1,), dtype=torch.bool)
        accumulator.record_layer(0, invalid_ids, active, valid, 1)
        accumulator.finish_step(1)
        self.assertTrue(accumulator.error)

        accumulator.on_heat_enabled()
        accumulator.record_layer(0, torch.tensor([[0]]), active, valid, 1)
        accumulator.finish_step(1)
        snapshot = accumulator.flush()
        self.assertIsNotNone(snapshot)
        self.assertFalse(snapshot.verified)
        self.assertTrue(snapshot.error)

        structural = DeviceHeatAccumulator(
            2,
            2,
            sync_period=1,
            top_k=1,
            max_rows=1,
            enabled=False,
            session_id="disabled-structural",
        )
        structural.record_layer(1, torch.tensor([[0]]), active, valid, 1)
        structural.finish_step(1)
        self.assertTrue(structural.error)
        structural.on_heat_enabled()
        structural.record_layer(0, torch.tensor([[0]]), active, valid, 1)
        structural.record_layer(1, torch.tensor([[0]]), active, valid, 1)
        structural.finish_step(1)
        structural_snapshot = structural.flush()
        self.assertIsNotNone(structural_snapshot)
        self.assertFalse(structural_snapshot.verified)
        self.assertTrue(structural_snapshot.error)

    def test_prefill_boundary_is_retained_until_first_decode_snapshot(self):
        kwargs = dict(
            num_layers=1,
            num_experts=3,
            hot_slots=1,
            decay=0.5,
            sync_period=5,
            hysteresis=0.0,
        )
        baseline = TierPolicy(**kwargs)
        imported = TierPolicy(**kwargs)
        accumulator = DeviceHeatAccumulator(
            1,
            3,
            decay=0.5,
            sync_period=5,
            top_k=1,
            max_rows=5,
            session_id="prefill",
        )

        prefill_routes = [[[2] for _ in range(5)]]
        prefill_activity = [[[True] for _ in range(5)]]
        prefill_valid = [True] * 5
        _record(accumulator, prefill_routes, prefill_activity, prefill_valid, 5)
        baseline.observe_step(prefill_routes, 5)
        snapshot = accumulator.flush()
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.tokens, 5)
        self.assertEqual(snapshot.forwards, 1)
        self.assertEqual(snapshot.route_total, 5)
        imported.import_device_snapshot(snapshot)
        self.assertIsNone(imported.plan_resync())
        self.assertEqual(imported._last_step_tokens, 5)
        accumulator.acknowledge_snapshot(snapshot)
        accumulator.rebase(
            tokens_total=imported.tokens_total,
            version=imported.version,
            last_sync_tokens=imported.last_sync_tokens,
        )

        decode_routes = [[[2]]]
        decode_activity = [[[True]]]
        decode_valid = [True]
        _record(accumulator, decode_routes, decode_activity, decode_valid, 1)
        baseline.observe_step(decode_routes, 1)
        snapshot = accumulator.snapshot()
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.tokens, 6)
        self.assertEqual(snapshot.forwards, 2)
        self.assertEqual(snapshot.route_total, 6)
        imported.import_device_snapshot(snapshot)
        self.assertEqual(imported.heat, baseline.heat)
        self.assertEqual(imported.tokens_total, baseline.tokens_total)
        self.assertEqual(imported._last_step_tokens, 1)
        self.assertEqual(imported.plan_resync(), baseline.plan_resync())

    def test_observation_only_consumes_due_boundaries_with_frozen_policy_clock(self):
        accumulator = DeviceHeatAccumulator(
            1,
            2,
            sync_period=4,
            top_k=1,
            max_rows=1,
            observation_only=True,
            session_id="observation-only-cadence",
        )
        snapshots = []
        for token in range(1, 13):
            _record(accumulator, [[[token % 2]]], [[[True]]], [True], 1)
            snapshot = accumulator.snapshot()
            if snapshot is None:
                self.assertFalse(accumulator.resync_due)
                continue
            snapshots.append(snapshot)
            self.assertEqual(snapshot.tokens, token)
            self.assertTrue(snapshot.resync_due)
            accumulator.acknowledge_snapshot(snapshot)
            accumulator.rebase(
                tokens_total=token,
                version=7,
                last_sync_tokens=0,
            )

        self.assertEqual([snapshot.tokens for snapshot in snapshots], [4, 8, 12])
        self.assertEqual(accumulator.base_tokens_total, 12)
        self.assertEqual(accumulator.base_version, 7)
        self.assertEqual(accumulator.base_last_sync_tokens, 0)
        self.assertFalse(accumulator.resync_due)

    def test_observation_only_prefill_flush_defers_due_until_first_decode(self):
        accumulator = DeviceHeatAccumulator(
            1,
            2,
            sync_period=5,
            top_k=1,
            max_rows=5,
            observation_only=True,
            session_id="observation-only-prefill",
        )
        _record(
            accumulator,
            [[[1] for _ in range(5)]],
            [[[True] for _ in range(5)]],
            [True] * 5,
            5,
        )
        prefill = accumulator.flush()
        self.assertIsNotNone(prefill)
        self.assertEqual(prefill.last_step_tokens, 5)
        self.assertTrue(prefill.resync_due)
        accumulator.acknowledge_snapshot(prefill)
        accumulator.rebase(tokens_total=5, version=3, last_sync_tokens=0)
        self.assertTrue(accumulator.resync_due)

        _record(accumulator, [[[1]]], [[[True]]], [True], 1)
        decode = accumulator.snapshot()
        self.assertIsNotNone(decode)
        self.assertEqual(decode.tokens, 6)
        self.assertEqual(decode.last_step_tokens, 1)
        accumulator.acknowledge_snapshot(decode)
        accumulator.rebase(tokens_total=6, version=3, last_sync_tokens=0)
        self.assertFalse(accumulator.resync_due)

    def test_observation_only_delayed_ack_keeps_newer_boundary_due(self):
        accumulator = DeviceHeatAccumulator(
            1,
            2,
            sync_period=4,
            top_k=1,
            max_rows=1,
            observation_only=True,
            session_id="observation-only-delayed-ack",
        )
        for _ in range(4):
            _record(accumulator, [[[0]]], [[[True]]], [True], 1)
        first = accumulator.snapshot()
        self.assertIsNotNone(first)
        self.assertEqual(first.tokens, 4)

        for _ in range(4):
            _record(accumulator, [[[1]]], [[[True]]], [True], 1)
        # The first pending snapshot was superseded before it was delivered;
        # acknowledging it must not consume the boundary crossed at token 8.
        accumulator.acknowledge_snapshot(first)
        self.assertTrue(accumulator.resync_due)
        second = accumulator.snapshot()
        self.assertIsNotNone(second)
        self.assertEqual(second.tokens, 8)
        self.assertGreater(second.sequence, first.sequence)
        accumulator.acknowledge_snapshot(second)
        self.assertFalse(accumulator.resync_due)

    def test_observation_only_setter_propagates_to_existing_accumulator(self):
        observer = DeviceObserver(
            num_layers=1,
            num_experts=2,
            sync_period=4,
            session_id="observation-only-setter",
        )
        observer.allocate("cpu", layers=1, top_k=1, max_tokens=1)
        self.assertFalse(observer.observation_only)
        self.assertFalse(observer.accumulator.observation_only)
        observer.set_observation_only()
        self.assertTrue(observer.observation_only)
        self.assertTrue(observer.accumulator.observation_only)
        observer.set_observation_only(False)
        self.assertFalse(observer.observation_only)
        self.assertFalse(observer.accumulator.observation_only)

    def test_import_preserves_dwell_pending_and_version_and_rejects_replay(self):
        policy = TierPolicy(
            1,
            3,
            1,
            sync_period=1,
            hysteresis=0.0,
            dwell_tokens=4,
            swaps_per_token=0.5,
            max_swaps_per_resync=1,
        )
        accumulator = DeviceHeatAccumulator(
            1,
            3,
            sync_period=1,
            top_k=1,
            max_rows=1,
            session_id="guards",
        )
        _record(accumulator, [[[2]]], [[[True]]], [True], 1)
        snapshot = accumulator.snapshot()
        self.assertIsNotNone(snapshot)
        dwell_before = policy.dwell_counts
        policy.import_device_snapshot(snapshot)
        self.assertEqual(policy.dwell_counts, dwell_before)
        self.assertEqual(policy.version, 0)
        with self.assertRaises(RuntimeError):
            policy.import_device_snapshot(snapshot)

        changed = DeviceSnapshot(
            heat=snapshot.heat,
            tokens=snapshot.tokens + 1,
            forwards=snapshot.forwards + 1,
            tokens_total=snapshot.tokens_total + 1,
            forwards_total=snapshot.forwards_total + 1,
            base_tokens_total=snapshot.base_tokens_total,
            base_version=1,
            base_last_sync_tokens=snapshot.base_last_sync_tokens,
            session_id=snapshot.session_id,
            sequence=snapshot.sequence + 1,
            last_step_tokens=1,
            resync_due=True,
        )
        with self.assertRaises(RuntimeError):
            policy.import_device_snapshot(changed)

        plan = policy.plan_resync()
        self.assertIsNotNone(plan)
        with self.assertRaises(RuntimeError):
            policy.import_device_snapshot(changed)
        policy.discard(plan)

    def test_invalid_ids_are_safe_and_snapshot_is_unverified(self):
        accumulator = DeviceHeatAccumulator(
            1,
            3,
            top_k=2,
            max_rows=1,
            session_id="invalid",
        )
        ids = torch.tensor([[-100, 100]], dtype=torch.int64)
        active = torch.ones((1, 2), dtype=torch.bool)
        valid = torch.ones((1,), dtype=torch.bool)
        accumulator.record_layer(0, ids, active, valid, 1)
        accumulator.finish_step(1)
        snapshot = accumulator.snapshot(force=True)
        self.assertIsNotNone(snapshot)
        self.assertFalse(snapshot.verified)
        self.assertTrue(snapshot.error)
        self.assertEqual(accumulator.heat.tolist(), [[0.0, 0.0, 0.0]])
        policy = TierPolicy(1, 3, 1)
        with self.assertRaises(RuntimeError):
            policy.import_device_snapshot(snapshot)

    def test_known_token_count_mismatch_is_delayed_to_snapshot_validation(self):
        accumulator = DeviceHeatAccumulator(
            1,
            2,
            top_k=1,
            max_rows=2,
            session_id="valid-count",
        )
        ids = torch.tensor([[0], [1]], dtype=torch.int64)
        active = torch.ones((2, 1), dtype=torch.bool)
        valid = torch.tensor([True, False])
        accumulator.record_layer(0, ids, active, valid, 2)
        # The runner says one true token; the device mask says one as well.
        accumulator.finish_step(1)
        good = accumulator.flush()
        self.assertIsNotNone(good)
        self.assertTrue(good.verified)
        accumulator.acknowledge_snapshot(good)

        accumulator.record_layer(0, ids, active, valid, 2)
        # A stale/incorrect host count is detected on-device and only exposed
        # when the explicit snapshot boundary reads the persistent flag.
        accumulator.finish_step(2)
        bad = accumulator.flush()
        self.assertIsNotNone(bad)
        self.assertFalse(bad.verified)
        self.assertTrue(bad.error)

    def test_dwell_hysteresis_budget_and_initial_scores_match_policy(self):
        config = dict(
            num_layers=3,
            num_experts=4,
            hot_slots=1,
            decay=0.75,
            sync_period=1,
            hysteresis=1.2,
            dwell_tokens=2,
            swaps_per_token=0.5,
            max_swaps_per_resync=1,
            initial_scores=[[5.0, 0.0, 0.0, 0.0]] * 3,
        )
        baseline = TierPolicy(**config)
        imported = TierPolicy(**config)
        accumulator = DeviceHeatAccumulator(
            3,
            4,
            decay=config["decay"],
            sync_period=1,
            initial_scores=config["initial_scores"],
            top_k=2,
            max_rows=2,
            session_id="policy-match",
        )
        steps = [
            (
                [[[1, 1], [1, 1]]] * 3,
                [[[True, False], [True, False]]] * 3,
                [True, False],
            ),
            (
                [[[1, 1], [2, 2]]] * 3,
                [[[True, False], [True, False]]] * 3,
                [True, False],
            ),
            ([[(2, 2), (2, 3)]] * 3, [[[True, True], [True, True]]] * 3, [True, False]),
            ([[(3, 3), (3, 3)]] * 3, [[[True, True], [True, True]]] * 3, [True, False]),
        ]
        for routes, activity, valid in steps:
            routes = [[list(row) for row in layer] for layer in routes]
            _record(accumulator, routes, activity, valid, sum(valid))
            weights = [
                [[1.0 if active else 0.0 for active in row] for row in layer]
                for layer in activity
            ]
            baseline.observe_step(
                routes,
                sum(valid),
                weights,
                valid_token_mask=valid,
            )
            snapshot = accumulator.snapshot()
            self.assertIsNotNone(snapshot)
            imported.import_device_snapshot(snapshot)
            self.assertEqual(imported.heat, baseline.heat)
            self.assertEqual(imported.dwell_counts, baseline.dwell_counts)
            self.assertEqual(imported.plan_resync(), baseline.plan_resync())
            plan = imported.plan_resync()
            if plan is not None:
                baseline_plan = baseline.plan_resync()
                imported.commit(plan)
                baseline.commit(baseline_plan)
                accumulator.acknowledge_snapshot(snapshot)
                accumulator.rebase(
                    tokens_total=imported.tokens_total,
                    version=imported.version,
                    last_sync_tokens=imported.last_sync_tokens,
                )
            else:
                accumulator.acknowledge_snapshot(snapshot)
                accumulator.rebase(
                    tokens_total=imported.tokens_total,
                    version=imported.version,
                    last_sync_tokens=imported.last_sync_tokens,
                )

    def test_observer_gate_and_finish_contract(self):
        observer = DeviceObserver(
            num_layers=1,
            num_experts=3,
            sync_period=1,
            session_id="observer",
        )
        observer.allocate("cpu", layers=1, top_k=2, max_tokens=2)
        ids = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int64)
        active = torch.tensor([[True, True], [False, False]])
        valid = torch.tensor([True, False])
        observer.record_layer(0, ids, active, valid, 2)
        self.assertIsInstance(observer.finish(2, 1, False, None, 3), Deferred)
        self.assertEqual(observer.accumulator.tokens_total, 0)
        observer.on_heat_enabled()
        observer.record_layer(0, ids, active, valid, 2)
        result = observer.finish(2, 1, True, None, 3)
        self.assertIsInstance(result, DeviceSnapshot)
        self.assertEqual(result.tokens, 1)
        self.assertEqual(result.tokens_total, 1)
        self.assertEqual(result.forwards, 1)
        self.assertEqual(result.forwards_total, 1)
        self.assertEqual(result.last_step_tokens, 1)
        self.assertEqual(result.route_total, 2)
        self.assertRaises(ValueError, observer.finish, 2, None, True, None, 3)

    def test_snapshot_counters_are_cumulative_and_heat_is_owned(self):
        accumulator = DeviceHeatAccumulator(
            1,
            2,
            decay=0.5,
            sync_period=50,
            top_k=1,
            max_rows=1,
            session_id="owned",
        )
        for expert in (0, 1):
            ids = torch.tensor([[expert]], dtype=torch.int64)
            active = torch.ones((1, 1), dtype=torch.bool)
            valid = torch.ones((1,), dtype=torch.bool)
            accumulator.record_layer(0, ids, active, valid, 1)
            accumulator.finish_step(1)
            snapshot = accumulator.flush()
            self.assertIsNotNone(snapshot)
            if expert == 0:
                first_heat = snapshot.heat
                self.assertEqual(snapshot.tokens, 1)
                self.assertEqual(snapshot.forwards, 1)
                self.assertEqual(snapshot.route_total, 1)
            else:
                self.assertEqual(snapshot.tokens, 2)
                self.assertEqual(snapshot.forwards, 2)
                self.assertEqual(snapshot.route_total, 2)
                self.assertEqual(first_heat, ((1.0, 0.0),))
            accumulator.acknowledge_snapshot(snapshot)

    def test_hot_map_counts_active_valid_duplicate_lanes_and_safe_ids(self):
        accumulator = DeviceHeatAccumulator(
            1,
            4,
            sync_period=50,
            top_k=6,
            max_rows=1,
            session_id="hot-map",
        )
        ids = torch.tensor([[0, 1, 1, 3, -1, 99]], dtype=torch.int64)
        active = torch.tensor([[True, True, False, True, True, True]])
        valid = torch.tensor([True])
        hot_map = torch.tensor([-1, 0, -1, 1], dtype=torch.int32)
        accumulator.record_layer(0, ids, active, valid, 1, hot_map)
        accumulator.finish_step(1)
        snapshot = accumulator.flush()
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.route_total, 3)
        self.assertEqual(snapshot.route_hot, 2)
        self.assertTrue(snapshot.route_hot_available)
        self.assertEqual(accumulator.route_hot, 2)

    def test_hot_map_is_referenced_and_missing_map_is_unavailable(self):
        accumulator = DeviceHeatAccumulator(
            1,
            2,
            sync_period=50,
            top_k=1,
            max_rows=1,
            session_id="hot-map-reference",
        )
        ids = torch.tensor([[0]], dtype=torch.int64)
        active = torch.ones((1, 1), dtype=torch.bool)
        valid = torch.ones((1,), dtype=torch.bool)
        hot_map = torch.tensor([0, -1], dtype=torch.int32)
        accumulator.record_layer(0, ids, active, valid, 1, hot_map)
        accumulator.finish_step(1)
        first = accumulator.flush()
        self.assertIsNotNone(first)
        self.assertEqual(first.route_hot, 1)
        self.assertTrue(first.route_hot_available)
        accumulator.acknowledge_snapshot(first)

        # A map is passed by reference and can be swapped in place between
        # graph replays without rebuilding the observer.
        hot_map.fill_(-1)
        accumulator.record_layer(0, ids, active, valid, 1, hot_map)
        accumulator.finish_step(1)
        second = accumulator.flush()
        self.assertIsNotNone(second)
        self.assertEqual(second.route_hot, 1)
        self.assertTrue(second.route_hot_available)
        accumulator.acknowledge_snapshot(second)

        unavailable = DeviceHeatAccumulator(
            1,
            2,
            sync_period=50,
            top_k=1,
            max_rows=1,
            session_id="hot-map-missing",
        )
        unavailable.record_layer(0, ids, active, valid, 1)
        unavailable.finish_step(1)
        missing = unavailable.flush()
        self.assertIsNotNone(missing)
        self.assertEqual(missing.route_hot, 0)
        self.assertFalse(missing.route_hot_available)

    def test_hot_map_availability_replays_without_python_record_layer(self):
        accumulator = DeviceHeatAccumulator(
            2,
            3,
            sync_period=50,
            top_k=1,
            max_rows=1,
            session_id="hot-map-replay",
        )
        ids = torch.tensor([[0]], dtype=torch.int64)
        active = torch.ones((1, 1), dtype=torch.bool)
        valid = torch.ones((1,), dtype=torch.bool)
        maps = [
            torch.tensor([0, -1, -1], dtype=torch.int32),
            torch.tensor([-1, 0, -1], dtype=torch.int32),
        ]
        for layer, hot_map in enumerate(maps):
            accumulator.record_layer(layer, ids, active, valid, 1, hot_map)
        accumulator.finish_step(1)
        first = accumulator.flush()
        self.assertIsNotNone(first)
        self.assertTrue(first.route_hot_available)
        accumulator.acknowledge_snapshot(first)

        # CUDA graph replay runs the captured device bookkeeping, while Python
        # does not call record_layer again.  Re-run only those device ops and
        # the captured counters to model that replay on CPU.
        for layer, hot_map in enumerate(maps):
            accumulator._record_hot_map_state(layer, hot_map)
        accumulator._step_route_total.fill_(2)
        accumulator._step_route_hot.fill_(2)
        accumulator._step_valid_tokens.fill_(1)
        accumulator._expected_layer.fill_(2)
        accumulator.finish_step(1)
        second = accumulator.flush()
        self.assertIsNotNone(second)
        self.assertTrue(second.route_hot_available)
        self.assertEqual(second.route_hot, 3)

    def test_hot_map_startup_reset_and_enabled_missing_map_invalidation(self):
        accumulator = DeviceHeatAccumulator(
            2,
            2,
            sync_period=50,
            top_k=1,
            max_rows=1,
            enabled=False,
            session_id="hot-map-startup",
        )
        ids = torch.tensor([[0]], dtype=torch.int64)
        active = torch.ones((1, 1), dtype=torch.bool)
        valid = torch.ones((1,), dtype=torch.bool)

        # A disabled startup/capture forward with no map must not poison the
        # first enabled observation.
        for layer in range(2):
            accumulator.record_layer(layer, ids, active, valid, 1)
        accumulator.finish_step(1)
        accumulator.on_heat_enabled()
        hot_map = torch.tensor([0, -1], dtype=torch.int32)
        for layer in range(2):
            accumulator.record_layer(layer, ids, active, valid, 1, hot_map)
        accumulator.finish_step(1)
        good = accumulator.flush()
        self.assertIsNotNone(good)
        self.assertTrue(good.route_hot_available)
        accumulator.acknowledge_snapshot(good)

        # Once an enabled boundary omits a layer map, its cumulative hit
        # counter can no longer claim complete coverage.
        accumulator.record_layer(0, ids, active, valid, 1, hot_map)
        accumulator.record_layer(1, ids, active, valid, 1)
        accumulator.finish_step(1)
        bad = accumulator.flush()
        self.assertIsNotNone(bad)
        self.assertFalse(bad.route_hot_available)

    def test_observer_passes_hot_map_to_device_accumulator(self):
        observer = DeviceObserver(
            num_layers=1,
            num_experts=3,
            sync_period=1,
            session_id="observer-hot-map",
        )
        observer.allocate("cpu", layers=1, top_k=2, max_tokens=1)
        observer.on_heat_enabled()
        ids = torch.tensor([[0, 2]], dtype=torch.int64)
        active = torch.ones((1, 2), dtype=torch.bool)
        valid = torch.ones((1,), dtype=torch.bool)
        hot_map = torch.tensor([0, -1, -1], dtype=torch.int32)
        observer.record_layer(0, ids, active, valid, 1, hot_map)
        result = observer.finish(1, 1, True, None, 3)
        self.assertIsInstance(result, DeviceSnapshot)
        self.assertEqual(result.route_hot, 1)
        self.assertTrue(result.route_hot_available)


if __name__ == "__main__":
    unittest.main()
