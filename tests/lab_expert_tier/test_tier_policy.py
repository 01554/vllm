# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only behavioral tests for the upstream-derived compact-bank policy.

Run: .venv/bin/python -m unittest discover -s tests/lab_expert_tier -v
No torch, CUDA, model download, or runtime process is required.
"""

import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "vllm" / "_lab_expert_tier")
)
from tier_policy import Swap, SwapPlan, TierPolicy


class TierPolicyTests(unittest.TestCase):
    def assert_partition(self, policy):
        for layer in range(policy.num_layers):
            hot, cold = policy.hot_to_expert[layer], policy.cold_to_expert[layer]
            self.assertEqual(len(hot), policy.hot_slots)
            self.assertEqual(len(cold), policy.num_experts - policy.hot_slots)
            self.assertEqual(sorted(hot + cold), list(range(policy.num_experts)))
            for expert in range(policy.num_experts):
                h = policy.expert_to_hot[layer][expert]
                c = policy.expert_to_cold[layer][expert]
                self.assertNotEqual(h >= 0, c >= 0)
                if h >= 0:
                    self.assertEqual(hot[h], expert)
                else:
                    self.assertEqual(cold[c], expert)

    def apply_physical_and_commit(self, policy, plan):
        """Simulate actual exclusive-bank copies using expert IDs as payloads."""
        hot = [list(row) for row in policy.hot_to_expert]
        cold = [list(row) for row in policy.cold_to_expert]
        for swap in plan.swaps:
            self.assertEqual(hot[swap.layer][swap.hot_slot], swap.old_expert)
            self.assertEqual(cold[swap.layer][swap.cold_slot], swap.new_expert)
            temp = hot[swap.layer][swap.hot_slot]  # D2H old -> one TEMP slice.
            hot[swap.layer][swap.hot_slot] = cold[swap.layer][swap.cold_slot]  # H2D.
            cold[swap.layer][swap.cold_slot] = temp  # CPU memcpy into vacated slot.
        policy.commit(plan)
        self.assertEqual(tuple(map(tuple, hot)), policy.hot_to_expert)
        self.assertEqual(tuple(map(tuple, cold)), policy.cold_to_expert)
        self.assert_partition(policy)

    def test_default_configuration_and_deterministic_partition(self):
        policy = TierPolicy(2, 5, 2)
        self.assertEqual(
            (
                policy.decay,
                policy.sync_period,
                policy.hysteresis,
                policy.dwell_tokens,
                policy.swaps_per_token,
            ),
            (0.999, 50, 1.3, 0, 1.0),
        )
        self.assertEqual(policy.hot_to_expert, ((0, 1), (0, 1)))
        self.assertEqual(policy.cold_to_expert, ((2, 3, 4), (2, 3, 4)))
        self.assertEqual(policy.swaps_total, 0)  # Initial fill is not migration.
        self.assertEqual(policy.max_swaps_per_resync, 0)
        self.assert_partition(policy)

    def test_initial_scores_rank_descending_and_break_ties_by_id(self):
        policy = TierPolicy(1, 5, 2, initial_scores=[[0, 5, 1, 5, 0]])
        self.assertEqual(policy.hot_to_expert, ((1, 3),))
        self.assertEqual(policy.cold_to_expert, ((0, 2, 4),))
        self.assert_partition(policy)

    def test_heat_decay_once_per_model_step_and_selection_not_probability(self):
        policy = TierPolicy(2, 3, 1, decay=0.5, initial_scores=[[4, 0, 0], [4, 0, 0]])
        policy.observe_step(
            [[[0], [0], [0]], [[0], [0], [0]]],
            3,
            [[[0.1], [0.2], [0.3]], [[1.0], [2.0], [3.0]]],
        )
        self.assertEqual(policy.heat, ((5.0, 0.0, 0.0), (5.0, 0.0, 0.0)))
        self.assertEqual(policy.tokens_total, 3)
        self.assertIsNone(policy.plan_resync())  # Prefill migration freeze.

    def test_vectorized_heat_update_matches_scalar_float_results(self):
        layers, experts = 48, 37
        initial = [
            [(layer + 1) * 0.1 + expert * 1e-12 for expert in range(experts)]
            for layer in range(layers)
        ]
        policy = TierPolicy(layers, experts, 10, decay=0.999, initial_scores=initial)
        expected = [row[:] for row in initial]
        for rows, tokens in ((5, 3), (1, 1), (4, 2)):
            routes, weights = [], []
            mask = [row < tokens for row in range(rows)]
            for layer in range(layers):
                layer_routes, layer_weights = [], []
                for row in range(rows):
                    layer_routes.append(
                        [(layer + row) % experts, -1, experts, (row + 3) % experts]
                    )
                    layer_weights.append([1.0, 1.0, 1.0, 0.0 if row % 2 else 2.0])
                routes.append(layer_routes)
                weights.append(layer_weights)

            policy.observe_step(routes, tokens, weights, mask)
            for layer, layer_routes in enumerate(routes):
                counts: dict[int, int] = {}
                for row, ids in enumerate(layer_routes):
                    if not mask[row]:
                        continue
                    for expert, weight in zip(ids, weights[layer][row]):
                        if 0 <= expert < experts and weight > 0:
                            counts[expert] = counts.get(expert, 0) + 1
                expected[layer] = [value * policy.decay for value in expected[layer]]
                for expert, count in counts.items():
                    expected[layer][expert] += count

        self.assertEqual(
            tuple(tuple(value.hex() for value in row) for row in policy.heat),
            tuple(tuple(value.hex() for value in row) for row in expected),
        )
        self.assertIs(type(policy.heat[0][0]), float)

    def test_subnormal_decay_ignores_numpy_error_mode_and_restores_it(self):
        minimum_subnormal = float.fromhex("0x0.0000000000001p-1022")
        policy = TierPolicy(
            1,
            2,
            1,
            decay=0.5,
            sync_period=1,
            initial_scores=[[minimum_subnormal, 0.0]],
        )
        previous = np.seterr(all="raise")
        try:
            policy.observe_step([[[1]]], 1)
            self.assertEqual(policy.heat, ((0.0, 1.0),))
            self.assertIs(type(policy.heat[0][0]), float)
            self.assertEqual(np.geterr()["under"], "raise")
        finally:
            np.seterr(**previous)
        self.assertEqual(np.geterr(), previous)

    def test_hysteresis_overflow_ignores_numpy_error_mode_and_restores_it(self):
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        policy = TierPolicy(
            1,
            2,
            1,
            decay=1.0,
            sync_period=1,
            hysteresis=maximum,
            initial_scores=[[2.0, 1.0]],
        )
        previous = np.seterr(all="raise")
        try:
            policy.observe_step([[[1, 1]]], 1)
            plan = policy.plan_resync()
            self.assertIsNotNone(plan)
            self.assertEqual(plan.swaps, ())
            self.assertEqual(policy.heat, ((2.0, 3.0),))
            self.assertIs(type(policy.heat[0][0]), float)
            self.assertEqual(np.geterr()["over"], "raise")
        finally:
            np.seterr(**previous)
        self.assertEqual(np.geterr(), previous)

    def test_model_global_cap_is_not_multiplied_by_48_layers(self):
        policy = TierPolicy(48, 3, 1, sync_period=1, hysteresis=0)
        policy.observe_step([[[2]]] * 48, 1)
        plan = policy.plan_resync()
        self.assertEqual(policy.tokens_total, 1)
        self.assertEqual(plan.swaps, (Swap(0, 0, 1, 0, 2),))
        self.apply_physical_and_commit(policy, plan)
        self.assertEqual(policy.swaps_total, 1)
        policy.observe_step([[[2]]] * 48, 1)
        plan = policy.plan_resync()
        self.assertEqual(plan.swaps, (Swap(1, 0, 1, 0, 2),))
        self.apply_physical_and_commit(policy, plan)
        self.assertEqual(policy.swaps_total, 2)

    def test_unused_model_token_credit_banks_across_prefill(self):
        policy = TierPolicy(8, 3, 1, sync_period=1, hysteresis=0)
        policy.observe_step([[[2]] * 5] * 8, 5)
        self.assertIsNone(policy.plan_resync())
        self.assertEqual(policy.swaps_total, 0)
        policy.observe_step([[[2]]] * 8, 1)
        plan = policy.plan_resync()
        self.assertEqual(len(plan.swaps), 6)
        self.assertEqual([s.layer for s in plan.swaps], list(range(6)))
        self.apply_physical_and_commit(policy, plan)
        self.assertEqual(policy.swaps_total, policy.tokens_total)

    def test_fractional_budget_preserves_upstream_preincrement_comparison(self):
        policy = TierPolicy(3, 2, 1, sync_period=1, hysteresis=0, swaps_per_token=0.5)
        policy.observe_step([[[1]]] * 3, 1)
        self.assertEqual(len(policy.plan_resync().swaps), 1)  # ceil(0.5), not floor.
        policy.commit(policy.plan_resync())
        policy.observe_step([[[1]]] * 3, 1)
        self.assertEqual(len(policy.plan_resync().swaps), 0)
        policy.commit(policy.plan_resync())

    def test_resync_cap_limits_banked_burst_across_all_layers_and_keeps_credit(self):
        policy = TierPolicy(
            8, 3, 1, sync_period=1, hysteresis=1.3, max_swaps_per_resync=2
        )
        policy.observe_step([[[2]] * 5] * 8, 5)
        self.assertIsNone(policy.plan_resync())
        for step, expected_layers in enumerate(
            ([0, 1], [2, 3], [4, 5], [6, 7]), start=1
        ):
            policy.observe_step([[[2]]] * 8, 1)
            plan = policy.plan_resync()
            self.assertEqual([swap.layer for swap in plan.swaps], expected_layers)
            self.assertEqual(policy.swaps_total, (step - 1) * 2)
            self.apply_physical_and_commit(policy, plan)
            self.assertEqual(policy.swaps_total, step * 2)
            self.assertEqual(policy.last_sync_tokens, 5 + step)
            self.assertLessEqual(policy.swaps_total, policy.tokens_total)
            if step == 1:
                # Reaching the cap must not skip dwell aging for later layers.
                self.assertEqual(policy.dwell_counts, ((0,), (0,)) + ((6,),) * 6)
        # Eight migrations over four decode steps use credit earned by prefill.
        self.assertEqual(policy.tokens_total, 9)

    def test_global_budget_can_be_stricter_than_positive_resync_cap(self):
        policy = TierPolicy(8, 3, 1, sync_period=1, max_swaps_per_resync=4)
        policy.observe_step([[[2]]] * 8, 1)
        self.assertEqual(len(policy.plan_resync().swaps), 1)
        self.apply_physical_and_commit(policy, policy.plan_resync())
        self.assertEqual(policy.swaps_total, 1)

    def test_fractional_global_budget_and_resync_cap_both_apply(self):
        policy = TierPolicy(
            8, 3, 1, sync_period=1, swaps_per_token=0.5, max_swaps_per_resync=2
        )
        policy.observe_step([[[2]] * 4] * 8, 4)
        policy.observe_step([[[2]]] * 8, 1)
        # Original budget allows ceil(2.5)=3; this plan is capped at two.
        self.assertEqual(len(policy.plan_resync().swaps), 2)
        self.apply_physical_and_commit(policy, policy.plan_resync())
        policy.observe_step([[[2]]] * 8, 1)
        # At six model tokens the cumulative limit is three, leaving only one.
        self.assertEqual(len(policy.plan_resync().swaps), 1)
        self.apply_physical_and_commit(policy, policy.plan_resync())
        self.assertEqual(policy.swaps_total, 3)
        policy.observe_step([[[2]]] * 8, 1)
        # Preserve the original fractional pre-increment comparison: ceil(3.5).
        self.assertEqual(len(policy.plan_resync().swaps), 1)
        self.apply_physical_and_commit(policy, policy.plan_resync())
        self.assertEqual(policy.swaps_total, 4)

    def test_explicit_zero_resync_cap_matches_omitted_default_with_banked_credit(self):
        for rate in (0.5, 1.0, 2.0):
            defaults = TierPolicy(8, 3, 1, sync_period=1, swaps_per_token=rate)
            zero = TierPolicy(
                8, 3, 1, sync_period=1, swaps_per_token=rate, max_swaps_per_resync=0
            )
            for tokens in (5, 1, 1, 1):
                for policy in (defaults, zero):
                    policy.observe_step([[[2]] * tokens] * 8, tokens)
                plans = [policy.plan_resync() for policy in (defaults, zero)]
                self.assertEqual(plans[0], plans[1])
                for policy, plan in zip((defaults, zero), plans):
                    if plan is not None:
                        self.apply_physical_and_commit(policy, plan)
                self.assertEqual(defaults.hot_to_expert, zero.hot_to_expert)
                self.assertEqual(defaults.cold_to_expert, zero.cold_to_expert)
                self.assertEqual(defaults.dwell_counts, zero.dwell_counts)
                self.assertEqual(defaults.swaps_total, zero.swaps_total)

    def test_resync_cap_does_not_disable_static_sync_or_grant_zero_rate_credit(self):
        for kwargs in ({"sync_period": 0}, {"sync_period": 1, "swaps_per_token": 0}):
            policy = TierPolicy(2, 3, 1, max_swaps_per_resync=2, **kwargs)
            policy.observe_step([[[2]]] * 2, 1)
            plan = policy.plan_resync()
            if plan is not None:
                self.assertEqual(plan.swaps, ())
                self.apply_physical_and_commit(policy, plan)
            self.assertEqual(policy.hot_to_expert, ((0,), (0,)))
            self.assertEqual(policy.tokens_total, 1)

    def test_resync_cap_requires_nonnegative_integer(self):
        for value in (-1, 1.0, 1.5, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TierPolicy(1, 3, 1, max_swaps_per_resync=value)

    def test_plan_does_not_publish_any_mapping_or_spend_budget_before_commit(self):
        policy = TierPolicy(1, 4, 2, sync_period=1, hysteresis=0)
        before = (
            policy.hot_to_expert,
            policy.cold_to_expert,
            policy.expert_to_hot,
            policy.expert_to_cold,
            policy.dwell_counts,
        )
        policy.observe_step([[[3]]], 1)
        plan = policy.plan_resync()
        self.assertIs(policy.plan_resync(), plan)
        self.assertEqual(
            before,
            (
                policy.hot_to_expert,
                policy.cold_to_expert,
                policy.expert_to_hot,
                policy.expert_to_cold,
                policy.dwell_counts,
            ),
        )
        self.assertEqual(
            (policy.swaps_total, policy.last_sync_tokens, policy.version), (0, 0, 0)
        )
        with self.assertRaises(RuntimeError):
            policy.observe_step([[[3]]], 1)
        self.apply_physical_and_commit(policy, plan)
        self.assertEqual(
            (policy.swaps_total, policy.last_sync_tokens, policy.version), (1, 1, 1)
        )
        self.assertIsNone(policy.plan_resync())
        with self.assertRaises(ValueError):
            policy.commit(plan)

    def test_unexecuted_discard_does_not_mutate_and_forged_or_foreign_plan_rejected(
        self,
    ):
        policy = TierPolicy(1, 2, 1, sync_period=1)
        policy.observe_step([[[1]]], 1)
        plan = policy.plan_resync()
        with self.assertRaises(ValueError):
            policy.commit(SwapPlan(plan.swaps, plan.tokens_total, plan.base_version))
        other = TierPolicy(1, 2, 1, sync_period=1)
        other.observe_step([[[1]]], 1)
        with self.assertRaises(ValueError):
            policy.commit(other.plan_resync())
        policy.discard(plan)
        self.assertEqual(policy.hot_to_expert, ((0,),))
        self.assertEqual((policy.swaps_total, policy.last_sync_tokens), (0, 0))
        with self.assertRaises(ValueError):
            policy.commit(plan)
        self.apply_physical_and_commit(policy, policy.plan_resync())

    def test_ordered_plan_simulates_multiple_swaps_reusing_one_hot_slot(self):
        policy = TierPolicy(
            1,
            4,
            2,
            decay=0,
            sync_period=1,
            hysteresis=0,
            swaps_per_token=2,
            initial_scores=[[0, 0, 2, 1]],
        )
        policy.observe_step([[[-1]]], 1)  # All scores now tie; top candidates are 0,1.
        plan = policy.plan_resync()
        self.assertEqual(plan.swaps, (Swap(0, 0, 0, 2, 0), Swap(0, 0, 1, 0, 1)))
        self.apply_physical_and_commit(policy, plan)
        self.assertEqual(policy.hot_to_expert, ((1, 3),))
        self.assertEqual(policy.cold_to_expert, ((2, 0),))

    def test_cadence_uses_crossed_integer_boundary_and_is_not_demand_fill(self):
        policy = TierPolicy(1, 3, 1, sync_period=3)
        for _ in range(2):
            policy.observe_step([[[2]]], 1)
            self.assertIsNone(policy.plan_resync())
            self.assertEqual(policy.hot_to_expert, ((0,),))
        policy.observe_step([[[2]]], 1)
        self.apply_physical_and_commit(policy, policy.plan_resync())
        self.assertEqual(policy.hot_to_expert, ((2,),))
        policy.observe_step([[[1]]], 1)
        self.assertIsNone(policy.plan_resync())

    def test_hysteresis_rejects_small_advantage_and_accepts_threshold_equality(self):
        policy = TierPolicy(
            1, 2, 1, decay=1, sync_period=1, hysteresis=1.5, initial_scores=[[2, 0]]
        )
        policy.observe_step([[[1, 1]]], 1)
        self.assertEqual(policy.plan_resync().swaps, ())
        policy.commit(policy.plan_resync())
        policy.observe_step([[[1]]], 1)
        self.assertEqual(policy.plan_resync().swaps, (Swap(0, 0, 0, 0, 1),))
        self.apply_physical_and_commit(policy, policy.plan_resync())

    def test_dwell_ages_by_tokens_at_sync_after_gate_and_new_slot_starts_at_zero(self):
        policy = TierPolicy(
            1, 3, 1, decay=0, sync_period=1, hysteresis=1.3, dwell_tokens=2
        )
        policy.observe_step([[[1]]], 1)
        self.apply_physical_and_commit(policy, policy.plan_resync())
        self.assertEqual(policy.dwell_counts, ((0,),))
        for expected_age in (1, 2):
            policy.observe_step([[[2]]], 1)
            self.assertEqual(policy.plan_resync().swaps, ())
            policy.commit(policy.plan_resync())
            self.assertEqual(policy.dwell_counts, ((expected_age,),))
        policy.observe_step([[[2]]], 1)
        self.apply_physical_and_commit(policy, policy.plan_resync())
        self.assertEqual(policy.hot_to_expert, ((2,),))
        self.assertEqual(policy.dwell_counts, ((0,),))

    def test_gate_off_also_disables_dwell_as_upstream_does(self):
        policy = TierPolicy(
            1, 3, 1, decay=0, sync_period=1, hysteresis=0, dwell_tokens=100
        )
        for expert in (1, 2):
            policy.observe_step([[[expert]]], 1)
            self.assertEqual(len(policy.plan_resync().swaps), 1)
            self.apply_physical_and_commit(policy, policy.plan_resync())
        self.assertEqual(policy.hot_to_expert, ((2,),))

    def test_zero_weight_and_padding_excluded_but_real_expert_zero_counts(self):
        policy = TierPolicy(1, 3, 1, sync_period=1)
        policy.observe_step(
            [[[0, 1, -1, 3], [2, 2, 2, 2]]],
            1,
            [[[0.5, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            [True, False],
        )
        self.assertEqual(policy.heat, ((1.0, 0.0, 0.0),))
        self.assertEqual(policy.tokens_total, 1)
        self.assertEqual(policy.plan_resync().swaps, ())

    def test_dummy_step_does_not_decay_gain_credit_or_trigger_sync(self):
        policy = TierPolicy(
            1, 3, 1, decay=0.5, sync_period=1, initial_scores=[[4, 0, 0]]
        )
        policy.observe_step([[[2]]], 0, [[[1.0]]], [False])
        self.assertEqual(policy.heat, ((4.0, 0.0, 0.0),))
        self.assertEqual(policy.tokens_total, 0)
        self.assertIsNone(policy.plan_resync())
        policy.observe_step([[]], 0)
        self.assertEqual(policy.heat, ((4.0, 0.0, 0.0),))

    def test_bad_last_layer_is_rejected_atomically(self):
        policy = TierPolicy(2, 3, 1, initial_scores=[[1, 0, 0], [1, 0, 0]])
        before = policy.heat
        for bad_weight in (float("nan"), float("inf"), -1.0):
            with self.subTest(bad_weight=bad_weight), self.assertRaises(ValueError):
                policy.observe_step([[[2]], [[2]]], 1, [[[1.0]], [[bad_weight]]])
            self.assertEqual(policy.heat, before)
            self.assertEqual(policy.tokens_total, 0)
        with self.assertRaises(ValueError):
            policy.observe_step([[[2]], [[1.5]]], 1)
        self.assertEqual(policy.heat, before)

    def test_inconsistent_shapes_masks_and_incomplete_model_steps_rejected(self):
        policy = TierPolicy(2, 3, 1)
        invalid_calls = [
            ([[[1]]], 1, None, None),
            ([[[1]], []], 1, None, None),
            ([[[1]], [[1]]], 0, None, None),
            ([[[1]], [[1]]], 1, None, [False]),
            ([[[1]], [[1]]], 1, None, [1]),
            ([[[1]], [[1]]], 1, [[[1]], [[]]], None),
            ([[[True]], [[1]]], 1, None, None),
        ]
        for args in invalid_calls:
            with self.subTest(args=args), self.assertRaises(ValueError):
                policy.observe_step(*args)
            self.assertEqual(policy.tokens_total, 0)

    def test_disabled_sync_and_zero_budget_keep_partition(self):
        for kwargs in ({"sync_period": 0}, {"sync_period": 1, "swaps_per_token": 0}):
            policy = TierPolicy(1, 3, 1, **kwargs)
            policy.observe_step([[[2]]], 1)
            plan = policy.plan_resync()
            if plan is not None:
                self.assertEqual(plan.swaps, ())
                self.apply_physical_and_commit(policy, plan)
            self.assertEqual(policy.hot_to_expert, ((0,),))
        for hot_slots in (0, 3):
            policy = TierPolicy(1, 3, hot_slots, sync_period=1)
            policy.observe_step([[[2]]], 1)
            plan = policy.plan_resync()
            if plan is not None:
                self.assertEqual(plan.swaps, ())
                self.apply_physical_and_commit(policy, plan)
            self.assert_partition(policy)

    def test_configuration_rejects_nonfinite_or_impossible_values(self):
        for kwargs in (
            {"decay": float("nan")},
            {"decay": 1.1},
            {"sync_period": -1},
            {"hysteresis": float("inf")},
            {"swaps_per_token": -1},
            {"dwell_tokens": True},
            {"initial_scores": [[0, -1, 0]]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TierPolicy(1, 3, 1, **kwargs)
        with self.assertRaises(ValueError):
            TierPolicy(1, 3, 4)

    def test_long_sequence_preserves_exclusive_payload_and_stable_untouched_slots(self):
        rng = random.Random(7)
        policy = TierPolicy(
            4, 12, 4, decay=0.9, sync_period=3, hysteresis=1.3, dwell_tokens=2
        )
        for step in range(100):
            tokens = 3 if step % 11 == 0 else 1
            routes = [
                [[rng.randrange(12), rng.randrange(12)] for _ in range(tokens)]
                for _ in range(4)
            ]
            policy.observe_step(routes, tokens)
            before = policy.hot_to_expert
            plan = policy.plan_resync()
            if plan is None:
                continue
            touched = {(swap.layer, swap.hot_slot) for swap in plan.swaps}
            self.apply_physical_and_commit(policy, plan)
            for layer, hot in enumerate(before):
                for slot, expert in enumerate(hot):
                    if (layer, slot) not in touched:
                        self.assertEqual(policy.hot_to_expert[layer][slot], expert)
            self.assertLessEqual(policy.swaps_total, policy.tokens_total)


if __name__ == "__main__":
    unittest.main()
