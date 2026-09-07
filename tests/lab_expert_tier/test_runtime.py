# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only invariants and real tensor byte/lifetime tests for the tier adapter."""

import gc
import importlib.util
import json
import math
import os
import sys
import unittest
import weakref
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

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

try:
    import torch
    from torch.multiprocessing.reductions import StorageWeakRef
except ImportError:
    torch = None


class InvariantTests(unittest.TestCase):
    def test_exact_six_tensor_budget_and_complementary_partition(self):
        slots, size = rt.uniform_slots(32 * 2**30, [2764808] * 48, 512)
        self.assertEqual((slots, size), (258, 34239382272))
        # Ten staging rows per layer come out of the same budget.
        staged_slots, staged_size = rt.uniform_slots(
            32 * 2**30, [2764808] * 48, 512, reserve=10
        )
        self.assertEqual((staged_slots, staged_size), (248, size))
        with self.assertRaises(ValueError):
            rt.uniform_slots(32 * 2**30, [2764808] * 48, 512, reserve=-1)
        with self.assertRaises(ValueError):
            rt.uniform_slots(10 * 2764808 * 48, [2764808] * 48, 512, reserve=10)
        h, c = tuple(range(258)) + (-1,) * 254, (-1,) * 258 + tuple(range(254))
        rt.validate_partition(h, c, 258, 254)
        self.assertEqual(size + 254 * 2764808 * 48, 67947921408)
        h2, c2 = rt.maps_after_swap(h, c, 0, 258, 0, 0)
        self.assertEqual((h2[0], c2[0], h2[258], c2[258]), (-1, 0, 0, -1))
        self.assertEqual(rt.maps_after_swap(h2, c2, 258, 0, 0, 0), (h, c))

    def test_partition_rejects_missing_duplicate_or_double_resident(self):
        for h, c in (
            ((0, -1), (0, -1)),
            ((0, 0), (-1, 0)),
            ((0, -2), (-1, 0)),
            ((0, -1), (-1, -1)),
        ):
            with self.assertRaises(AssertionError):
                rt.validate_partition(h, c, 1, 1)
        with self.assertRaises(AssertionError):
            rt.maps_after_swap((0, -1), (-1, 0), 1, 0, 0, 0)

    def test_padding_and_zero_weight_activity_are_preserved(self):
        packed = [[[0, 2, 1, 0, 1], [-1, -1, 0, 0, 0]]] * 2
        routes, activity, mask, tokens = rt.unpack_routes(packed, 4)
        self.assertEqual(routes[0], [[0, 2], [-1, -1]])
        self.assertEqual(activity[0], [[1, 0], [0, 0]])
        self.assertEqual((mask, tokens), ([True, False], 1))
        self.assertEqual(rt.unpack_routes([[[-1, -1, 0, 0, 0]]], 4)[-1], 0)

    def test_padded_positive_and_sentinel_ids_keep_route_shape(self):
        packed = [
            [[2, 2, 1, 0, 1], [3, -1, 0, 0, 0]],
            [[2, 2, 1, 0, 1], [3, -1, 0, 0, 0]],
        ]
        routes, activity, mask, tokens = rt.unpack_routes(packed, 4)
        self.assertEqual(routes[0], [[2, 2], [3, -1]])
        self.assertEqual(activity[0], [[1, 0], [0, 0]])
        self.assertEqual((mask, tokens), ([True, False], 1))

    def test_invalid_real_sentinel_and_bad_model_record_fail(self):
        packed: list[Any]
        for packed in (
            [[[-1, 1, 1]]],
            [[[4, 1, 1]]],
            [[[-2, 1, 0]]],
            [[[0, 1, 1]], [[-1, 1, 1]]],
            [[[0, 1, 1]], [[0, 1, 0]]],
            [[[0, 3, 1]]],
            [[[0, 1, 2]]],
            [],
            [[[]]],
        ):
            with self.subTest(packed=packed), self.assertRaises(ValueError):
                rt.unpack_routes(packed, 4)

    def test_settings_reject_noop_or_unknown_knobs(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(rt.Settings.from_env())
        with patch.dict(os.environ, {rt.PREFIX + "GIB": "32"}, clear=True):
            self.assertEqual(rt.Settings.from_env(), rt.Settings(32 * 2**30))
        for suffix, value in (
            ("LRU", "1"),
            ("GIB", "-1"),
            ("VERIFY_INIT", "2"),
            ("STATS_EVERY", "0"),
            ("TEMP_SLOTS", "0"),
            ("SPLIT", "single"),
            ("STAGING", "2"),
        ):
            env = {rt.PREFIX + "GIB": "32", rt.PREFIX + suffix: value}
            with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                rt.Settings.from_env()

    def test_tuning_settings_are_validated_and_wired_to_policy(self):
        controls = {
            "GIB": "32",
            "SYNC_TOKENS": "10",
            "SWAPS_PER_TOKEN": "0.25",
            "DECAY": "0.9",
            "HYSTERESIS": "0",
            "DWELL_TOKENS": "12",
            "MAX_SWAPS_PER_RESYNC": "3",
            "STATS_EVERY": "1",
            "TEMP_SLOTS": "4",
            "SPLIT": "modular",
            "STAGING": "1",
        }
        with patch.dict(
            os.environ, {rt.PREFIX + k: v for k, v in controls.items()}, clear=True
        ):
            settings = rt.Settings.from_env()
        coordinator = rt.TierCoordinator(
            [SimpleNamespace(num_experts=4, hot_slots=2)], settings, {}
        )
        self.assertEqual((settings.stats_every, settings.temp_slots), (1, 4))
        self.assertEqual((settings.split, settings.staging), ("modular", True))
        with patch.dict(os.environ, {rt.PREFIX + "GIB": "32"}, clear=True):
            defaults = rt.Settings.from_env()
        self.assertEqual((defaults.split, defaults.staging), ("fused", False))
        expected = {
            "sync_period": 10,
            "swaps_per_token": 0.25,
            "decay": 0.9,
            "hysteresis": 0.0,
            "dwell_tokens": 12,
            "max_swaps_per_resync": 3,
        }
        self.assertEqual(settings.policy_kwargs(), expected)
        for key, expected_value in expected.items():
            self.assertEqual(getattr(coordinator.policy, key), expected_value)
        for key, value in (
            ("SYNC_TOKENS", "-1"),
            ("SYNC_TOKENS", "1.5"),
            ("DWELL_TOKENS", "-1"),
            ("MAX_SWAPS_PER_RESYNC", "-1"),
            ("SWAPS_PER_TOKEN", "nan"),
            ("SWAPS_PER_TOKEN", "-1"),
            ("DECAY", "1.001"),
            ("DECAY", "-0.1"),
            ("HYSTERESIS", "inf"),
            ("HYSTERESIS", "-1"),
        ):
            with (
                self.subTest(key=key, value=value),
                patch.dict(
                    os.environ,
                    {rt.PREFIX + "GIB": "32", rt.PREFIX + key: value},
                    clear=True,
                ),
                self.assertRaises(ValueError),
            ):
                rt.Settings.from_env()

    def test_stats_snapshot_has_explicit_zero_counters_and_effective_config(self):
        coordinator = rt.TierCoordinator(
            [SimpleNamespace(num_experts=4, hot_slots=2)],
            rt.Settings(32 * 2**30, sync_tokens=0),
            {},
        )
        with (
            patch.object(rt.LOGGER, "warning") as logger,
            patch.object(rt.time, "time_ns", return_value=1234),
        ):
            coordinator.report()
        result = json.loads(logger.call_args.args[1])
        self.assertEqual(
            (result["route_hot"], result["route_total"], result["per_layer_swaps"]),
            (0, 0, [0]),
        )
        self.assertEqual(
            (result["tokens_total"], result["version"], result["timestamp_ns"]),
            (0, 0, 1234),
        )
        self.assertEqual(result["policy_cpu_seconds"], 0)
        self.assertEqual(result["migration_wall_seconds"], 0)
        self.assertFalse(result["heat_enabled"])
        self.assertTrue(result["route_counts_before_migration"])
        self.assertEqual(result["policy_config"]["sync_period"], 0)
        self.assertEqual(result["settings"]["capacity_bytes"], 32 * 2**30)

    def test_registry_never_uses_tensor_equality_or_attaches_owner(self):
        class Parameter:
            __hash__: Any = None

            def __eq__(self, other):
                raise AssertionError("Tensor equality is not identity")

        with (
            patch.object(rt, "_CPU_SOURCES", {}),
            patch.object(rt, "_CAPTURE_COUNT", 0),
            patch.dict(os.environ, {rt.PREFIX + "GIB": "32"}, clear=True),
        ):
            p, first, second = Parameter(), object(), object()
            rt.capture_cpu_source(p, first)
            stale = rt._CPU_SOURCES[id(p)][0]
            rt.capture_cpu_source(p, second)
            stale.__callback__(stale)
            self.assertIs(rt._get_cpu_source(p), second)
            self.assertEqual(p.__dict__, {})
            del p
            gc.collect()
            self.assertFalse(rt._CPU_SOURCES)

    def test_ram_temp_order_waits_before_host_overwrite(self):
        events: list[str | tuple[str, str, bool]] = []

        class Row:
            def __init__(self, name):
                self.name = name

            def copy_(self, source, non_blocking=False):
                events.append((source.name, self.name, non_blocking))

        hot = {name: [Row("hot:" + name)] for name in rt.TENSORS}
        cold = {name: [Row("cold:" + name)] for name in rt.TENSORS}
        temp = {name: Row("temp:" + name) for name in rt.TENSORS}
        rt.swap_tensor_rows(hot, cold, temp, 0, 0, lambda: events.append("wait"))
        self.assertEqual(events[6], "wait")
        self.assertEqual(events[13], "wait")
        self.assertEqual(len(events), 20)
        for i, name in enumerate(rt.TENSORS):
            self.assertEqual(events[i], ("hot:" + name, "temp:" + name, True))
            self.assertEqual(events[7 + i], ("cold:" + name, "hot:" + name, True))
            self.assertEqual(events[14 + i], ("temp:" + name, "cold:" + name, False))

    def test_stream_handoff_waits_only_at_complete_model_boundaries(self):
        layer = SimpleNamespace(num_experts=4, hot_slots=2)
        coordinator = rt.TierCoordinator([layer], rt.Settings(32 * 2**30), {})
        first = SimpleNamespace(cuda_stream=11, synchronize=Mock())
        second = SimpleNamespace(cuda_stream=22, synchronize=Mock())
        coordinator.adopt_stream(first, True)
        coordinator.adopt_stream(first, True)
        first.synchronize.assert_not_called()
        coordinator.adopt_stream(second, True)
        first.synchronize.assert_called_once_with()
        self.assertIs(coordinator.stream, second)
        self.assertEqual(coordinator.stream_id, 22)
        coordinator.adopt_stream(second, True)
        second.synchronize.assert_not_called()

    def test_stream_handoff_rejects_midforward_and_keeps_previous_on_wait_failure(self):
        layer = SimpleNamespace(num_experts=4, hot_slots=2)
        coordinator = rt.TierCoordinator([layer], rt.Settings(32 * 2**30), {})
        first = SimpleNamespace(cuda_stream=11, synchronize=Mock())
        second = SimpleNamespace(cuda_stream=22, synchronize=Mock())
        coordinator.adopt_stream(first, True)
        coordinator.recorded = 1
        with self.assertRaises(NotImplementedError):
            coordinator.adopt_stream(second, False)
        with self.assertRaises(NotImplementedError):
            coordinator.adopt_stream(second, True)
        coordinator.adopt_stream(first, False)
        first.synchronize.assert_not_called()
        coordinator.recorded = 0
        with self.assertRaises(NotImplementedError):
            coordinator.adopt_stream(second, False)
        first.synchronize.side_effect = RuntimeError("failed previous stream")
        with self.assertRaises(RuntimeError):
            coordinator.adopt_stream(second, True)
        self.assertIs(coordinator.stream, first)
        self.assertEqual(coordinator.stream_id, 11)

    def test_heat_enable_requires_successful_complete_startup_and_waits_once(self):
        layer = SimpleNamespace(num_experts=4, hot_slots=2)
        coordinator = rt.TierCoordinator([layer], rt.Settings(32 * 2**30), {})
        self.assertFalse(coordinator.heat_enabled)
        stream = SimpleNamespace(cuda_stream=11, synchronize=Mock())
        coordinator.adopt_stream(stream, True)
        coordinator.recorded = 1
        with self.assertRaises(RuntimeError):
            coordinator.enable_heat()
        coordinator.recorded = 0
        # The last direct warmup/capture forward is startup work, not an error.
        coordinator.forward_rows = 3
        coordinator.poisoned = True
        with self.assertRaises(RuntimeError):
            coordinator.enable_heat()
        coordinator.poisoned = False
        stream.synchronize.side_effect = RuntimeError("startup CUDA error")
        with self.assertRaises(RuntimeError):
            coordinator.enable_heat()
        self.assertFalse(coordinator.heat_enabled)
        stream.synchronize.side_effect = None
        stream.synchronize.reset_mock()
        coordinator.stats["model_forwards"] = 4
        with patch.object(rt.LOGGER, "warning") as log:
            coordinator.enable_heat()
        stream.synchronize.assert_called_once_with()
        self.assertTrue(coordinator.heat_enabled)
        self.assertEqual(dict(coordinator.stats), {})
        self.assertIsNone(coordinator.forward_rows)
        self.assertIn('"dropped_startup_records": 1', log.call_args.args[1])
        self.assertEqual(
            (coordinator.policy.tokens_total, coordinator.policy.swaps_total), (0, 0)
        )
        self.assertIn('"model_forwards": 4', log.call_args.args[1])
        with self.assertRaises(RuntimeError):
            coordinator.enable_heat()

    def test_model_heat_hook_disabled_noop_and_missing_enabled_model_fails(self):
        with patch.dict(os.environ, {}, clear=True):
            rt.enable_model_heat(SimpleNamespace())
        with patch.dict(os.environ, {rt.PREFIX + "GIB": "32"}, clear=True):
            with self.assertRaises(RuntimeError):
                rt.enable_model_heat(SimpleNamespace())
            coordinator = SimpleNamespace(enable_heat=Mock())
            rt.enable_model_heat(
                SimpleNamespace(_lab_expert_tier_coordinator=coordinator)
            )
            coordinator.enable_heat.assert_called_once_with()


@unittest.skipIf(torch is None, "CPU torch is not installed")
class TensorTests(unittest.TestCase):
    def test_six_tensor_partition_swap_and_restore_are_byte_exact(self):
        dtypes = (
            torch.int32,
            torch.int32,
            torch.uint8,
            torch.uint8,
            torch.float32,
            torch.float32,
        )
        full = {
            name: torch.arange(4 * (i + 1)).reshape(4, i + 1).to(dtype)
            for i, (name, dtype) in enumerate(zip(rt.TENSORS, dtypes))
        }
        hot = {name: t[:2].clone() for name, t in full.items()}
        cold = {name: t[2:].clone() for name, t in full.items()}
        temp = {name: torch.empty_like(t[0]) for name, t in full.items()}
        self.assertEqual(
            sum(t.nbytes for t in full.values()),
            sum(t.nbytes for t in hot.values()) + sum(t.nbytes for t in cold.values()),
        )
        waits = []
        rt.swap_tensor_rows(hot, cold, temp, 0, 1, lambda: waits.append(1))
        self.assertEqual(len(waits), 2)
        for name in rt.TENSORS:
            torch.testing.assert_close(hot[name][0], full[name][3], rtol=0, atol=0)
            torch.testing.assert_close(cold[name][1], full[name][0], rtol=0, atol=0)
        rt.swap_tensor_rows(hot, cold, temp, 0, 1, lambda: None)
        for name in rt.TENSORS:
            torch.testing.assert_close(
                torch.cat((hot[name], cold[name])), full[name], rtol=0, atol=0
            )

    def test_compaction_frees_all_raw_storage_with_reload_metadata_alive(self):
        with (
            patch.object(rt, "_CPU_SOURCES", {}),
            patch.object(rt, "_CAPTURE_COUNT", 0),
            patch.dict(os.environ, {rt.PREFIX + "GIB": "32"}, clear=True),
        ):
            layer = torch.nn.Module()
            refs, storage_refs, metadata, hot, cold = [], [], [], {}, {}
            for i, name in enumerate(rt.TENSORS):
                source = torch.arange(24, dtype=torch.int32).reshape(4, 6).clone()
                parameter = torch.nn.Parameter(source, requires_grad=False)
                parameter._vllm_is_uva_offloaded = True
                setattr(layer, name, parameter)
                rt.capture_cpu_source(parameter, source)
                meta = parameter.data.to("meta")
                meta.__class__ = parameter.__class__
                meta.__dict__ = parameter.__dict__.copy()
                metadata.append(meta)
                refs.append(weakref.ref(source))
                storage_refs.append(StorageWeakRef(source.untyped_storage()))
                hot[name], cold[name] = source[:2].clone(), source[2:].clone()
            # The method's quant/kernel roots deliberately retain old Parameters.
            old = {name: getattr(layer, name) for name in rt.TENSORS}
            method = SimpleNamespace(
                moe_kernel=SimpleNamespace(weights=old), moe_quant_config=old
            )
            hot_quant = dict(hot)
            hot_kernel = SimpleNamespace(weights=hot_quant)
            rt.replace_full_source_references(layer, method, hot, hot_kernel, hot_quant)
            del source, parameter, old
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))
            self.assertTrue(all(ref.expired() for ref in storage_refs))
            self.assertFalse(rt._CPU_SOURCES)
            self.assertEqual(len(metadata), 6)
            self.assertIs(method.moe_kernel, hot_kernel)
            for name in rt.TENSORS:
                self.assertEqual(getattr(layer, name).shape[0], 2)
                self.assertEqual(cold[name].shape[0], 2)

    def test_split_preserves_hot_output_across_workspace_reuse(self):
        tier = object.__new__(rt.TierLayer)
        tier.settings = rt.Settings(32 * 2**30, split="modular")
        tier.hot_kernel, tier.cold_kernel = "hot", "cold"
        tier.hot, tier.cold, tier.hot_map, tier.cold_map = {}, {}, object(), object()
        workspace = torch.empty(2, 3, dtype=torch.bfloat16)

        def call(kernel, tensors, mapping, x, weights, ids):
            workspace.fill_(2 if kernel == "hot" else 5)
            return workspace

        tier.call = call
        output = tier.split(None, None, None)
        self.assertTrue(torch.equal(output, torch.full_like(output, 7)))

    def test_numerical_guard_rejects_wrong_or_nonfinite_outputs(self):
        good = torch.ones(2, 3, dtype=torch.bfloat16)
        self.assertTrue(
            rt.compare_outputs(good, good.clone(), "layer", "test")["finite"]
        )
        with self.assertRaises(AssertionError):
            rt.compare_outputs(good * 2, good, "layer", "bad")
        with self.assertRaises(AssertionError):
            rt.compare_outputs(good * float("nan"), good, "layer", "nan")

    def make_coordinator(self, num_layers=2, settings=None):
        layers = [
            SimpleNamespace(
                index=i,
                num_experts=4,
                hot_slots=2,
                hot_map_host=(0, 1, -1, -1),
                cold_map_host=(-1, -1, 0, 1),
            )
            for i in range(num_layers)
        ]
        coordinator = rt.TierCoordinator(
            layers, settings or rt.Settings(32 * 2**30), {}
        )
        coordinator.allocate_records(torch.device("cpu"), 2, 4)
        return coordinator

    @staticmethod
    def replay(coordinator, record):
        """A CUDA Graph replay: the static records change, no layer Python runs."""
        rows = record.shape[0]
        coordinator.records[:rows] = record[:, None, :]
        coordinator.finish_forward(rows)

    def test_static_partition_collects_heat_and_route_counts_without_migrations(self):
        coordinator = self.make_coordinator(
            settings=rt.Settings(32 * 2**30, sync_tokens=0)
        )
        coordinator.enable_heat()
        # Only expert 2 has positive weight; expert 0 and the padded row do not count.
        record = torch.tensor([[0, 2, 0, 1, 1], [-1, -1, 1, 1, 0]], dtype=torch.int32)
        for _ in range(51):
            self.replay(coordinator, record)
        self.assertEqual(coordinator.policy.tokens_total, 51)
        self.assertEqual(coordinator.stats["replayed_forwards"], 51)
        self.assertGreater(coordinator.policy.heat[0][2], 0)
        self.assertEqual(
            (coordinator.stats["route_hot"], coordinator.stats["route_total"]), (0, 102)
        )
        self.assertEqual(
            (coordinator.stats["swaps"], coordinator.stats["resyncs"]), (0, 0)
        )
        self.assertEqual(coordinator.policy.expert_to_hot[0], (0, 1, -1, -1))

    def test_migration_counts_prior_hot_placement_and_separates_completed_transfer_time(
        self,
    ):
        settings = rt.Settings(
            32 * 2**30,
            sync_tokens=1,
            swaps_per_token=2,
            decay=1,
            hysteresis=1,
            max_swaps_per_resync=1,
        )
        coordinator = self.make_coordinator(settings=settings)
        coordinator.enable_heat()
        events: list[Any] = []
        stream = SimpleNamespace(
            cuda_stream=7, synchronize=lambda: events.append("synchronized")
        )
        for layer in coordinator.layers:
            layer.row_bytes = 12
            layer.hot, layer.cold_cpu = {"h": layer.index}, {"c": layer.index}
            layer.publish_maps = Mock(side_effect=lambda: events.append("published"))

            def stage_swap(old, new, hot_slot, cold_slot, layer=layer):
                # Accounting must reflect where the just-completed forward ran.
                self.assertEqual(coordinator.stats["route_hot"], 0)
                self.assertEqual(coordinator.stats["route_total"], 4)
                layer.hot_map_host, layer.cold_map_host = rt.maps_after_swap(
                    layer.hot_map_host,
                    layer.cold_map_host,
                    old,
                    new,
                    hot_slot,
                    cold_slot,
                )
                events.append((layer.index, old, new))

            layer.stage_swap = stage_swap

        def wave(items, synchronize):
            events.append(
                ("wave", [(h["h"], c["c"], hs, cs) for h, c, _, hs, cs in items])
            )
            synchronize()

        record = torch.tensor([[2, 2, 1, 1, 1]], dtype=torch.int32)
        with (
            patch.object(rt, "_current_stream", return_value=stream),
            patch.object(rt, "swap_tensor_rows_wave", wave),
            patch.object(
                rt.time, "perf_counter", side_effect=[10, 11, 20, 22, 30, 35, 40, 43]
            ),
        ):
            self.replay(coordinator, record)
        # Boundary D2H wait, staged maps, the wave's wait, publication, then
        # the completed-migration wait.
        self.assertEqual(
            events,
            [
                "synchronized",
                (0, 0, 2),
                ("wave", [(0, 0, 0, 0)]),
                "synchronized",
                "published",
                "synchronized",
            ],
        )
        self.assertEqual(
            (coordinator.stats["migration_waves"], coordinator.stats["max_wave_swaps"]),
            (1, 1),
        )
        self.assertEqual(coordinator.per_layer_swaps, [1, 0])
        self.assertEqual(
            (
                coordinator.stats["swaps"],
                coordinator.stats["h2d_bytes"],
                coordinator.stats["d2h_bytes"],
                coordinator.stats["host_copy_bytes"],
            ),
            (1, 12, 12, 12),
        )
        self.assertEqual(coordinator.stats["policy_observe_seconds"], 1)
        self.assertEqual(coordinator.stats["policy_plan_seconds"], 2)
        self.assertEqual(coordinator.stats["policy_commit_seconds"], 3)
        self.assertEqual(coordinator.stats["migration_wall_seconds"], 5)
        self.assertEqual(
            (coordinator.policy.tokens_total, coordinator.policy.version), (1, 1)
        )
        with patch.object(rt.LOGGER, "warning") as logger:
            coordinator.report()
        snapshot = json.loads(logger.call_args.args[1])
        self.assertEqual(snapshot["policy_cpu_seconds"], 6)
        self.assertEqual(snapshot["per_layer_swaps"], [1, 0])

    def test_model_boundary_updates_heat_once_and_padding_never_ages(self):
        coordinator = self.make_coordinator(48)
        coordinator.enable_heat()
        record = torch.tensor([[0, 2, 1, 0, 1], [-1, -1, 0, 0, 0]], dtype=torch.int32)
        self.replay(coordinator, record)
        self.assertEqual(coordinator.policy.tokens_total, 1)
        self.assertEqual(coordinator.policy.heat[0], (1, 0, 0, 0))
        self.replay(coordinator, torch.tensor([[-1, -1, 0, 0, 0]], dtype=torch.int32))
        self.assertEqual(coordinator.policy.tokens_total, 1)
        self.assertEqual(coordinator.stats["ignored_synthetic_forwards"], 1)

    def test_builtin_warmup_real_mask_cannot_pollute_heat_or_swap_budget(self):
        coordinator = self.make_coordinator(48)
        # Positive IDs, active weights, real-looking mask: exactly why the
        # MRv2 builtin warmup cannot be identified from ForwardContext alone.
        record = torch.tensor([[2, 3, 1, 1, 1]], dtype=torch.int32)
        for _ in range(50):
            self.replay(coordinator, record)
        self.assertEqual(coordinator.policy.tokens_total, 0)
        self.assertEqual(coordinator.policy.swaps_total, 0)
        self.assertTrue(
            all(all(value == 0 for value in row) for row in coordinator.policy.heat)
        )
        self.assertEqual(coordinator.recorded, 0)
        self.assertEqual(coordinator.stats["ignored_startup_forwards"], 50)
        coordinator.enable_heat()
        self.replay(coordinator, record)
        self.assertEqual(coordinator.policy.tokens_total, 1)
        self.assertEqual(coordinator.policy.heat[0], (0, 0, 1, 1))

    def test_device_guard_accepts_padding_rejects_real_negative(self):
        coordinator = self.make_coordinator(1)
        tier = coordinator.layers[0]
        tier.device = torch.device("cpu")
        tier.method = SimpleNamespace(moe=SimpleNamespace(experts_per_token=2))
        x = torch.ones(2, 3, dtype=torch.bfloat16)
        weights = torch.ones(2, 2)
        mask = torch.tensor([False, True])
        module = SimpleNamespace(
            get_forward_context=lambda: SimpleNamespace(is_padding=mask)
        )
        with (
            patch.dict(sys.modules, {"vllm.forward_context": module}),
            patch.object(
                torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=1),
            ),
        ):
            coordinator.begin_layer(tier, x, weights, torch.tensor([[0, 2], [-1, -1]]))
            self.assertEqual((coordinator.recorded, coordinator.forward_rows), (1, 2))
            self.assertEqual(
                coordinator.records[:2, 0].tolist(),
                [[0, 2, 1, 1, 1], [-1, -1, 1, 1, 0]],
            )
            coordinator.recorded = 0
            with self.assertRaises(RuntimeError):
                coordinator.begin_layer(
                    tier, x, weights, torch.tensor([[-1, 2], [-1, -1]])
                )
            # Heat was off, so the unfinished first forward was startup work.
            self.assertEqual(coordinator.stats["dropped_startup_records"], 1)

    def forward_context(self, coordinator, mask):
        tier = coordinator.layers[0]
        tier.device = torch.device("cpu")
        tier.method = SimpleNamespace(moe=SimpleNamespace(experts_per_token=2))
        for layer in coordinator.layers[1:]:
            layer.device, layer.method = tier.device, tier.method
        module = SimpleNamespace(
            get_forward_context=lambda: SimpleNamespace(is_padding=mask)
        )
        return patch.dict(sys.modules, {"vllm.forward_context": module})

    def test_map_publication_keeps_device_addresses_for_captured_graphs(self):
        tier = object.__new__(rt.TierLayer)
        tier.device = torch.device("cpu")
        tier.hot_slots, tier.cold_slots = 2, 2
        tier.hot_map_host, tier.cold_map_host = (0, 1, -1, -1), (-1, -1, 0, 1)
        tier.hot_map = tier.cold_map = None
        tier.publish_maps()
        hot_ptr, cold_ptr = tier.hot_map.data_ptr(), tier.cold_map.data_ptr()
        tier.hot_map_host, tier.cold_map_host = rt.maps_after_swap(
            tier.hot_map_host, tier.cold_map_host, 0, 2, 0, 0
        )
        tier.publish_maps()
        self.assertEqual(
            (tier.hot_map.data_ptr(), tier.cold_map.data_ptr()), (hot_ptr, cold_ptr)
        )
        self.assertEqual(tier.hot_map.tolist(), [-1, 1, 0, -1])
        self.assertEqual(tier.cold_map.tolist(), [0, -1, -1, 1])
        tier.hot_map_host = (0, 0, -1, -1)
        with self.assertRaises(AssertionError):
            tier.publish_maps()

    def test_eager_forward_records_static_buffer_and_runner_finishes_it(self):
        coordinator = self.make_coordinator()
        coordinator.enable_heat()
        x = torch.ones(2, 3, dtype=torch.bfloat16)
        weights = torch.tensor([[0.5, 0.5], [0.0, 0.0]])
        ids = torch.tensor([[2, 3], [-1, -1]])
        with self.forward_context(coordinator, torch.tensor([False, True])):
            for tier in coordinator.layers:
                coordinator.begin_layer(tier, x, weights, ids)
                coordinator.end_layer(tier)
            self.assertEqual((coordinator.recorded, coordinator.forward_rows), (0, 2))
            with self.assertRaises(ValueError):
                coordinator.finish_forward(3)
            self.assertTrue(coordinator.poisoned)
            coordinator.poisoned = False
            coordinator.forward_rows = 2
            coordinator.finish_forward(2)
        self.assertIsNone(coordinator.forward_rows)
        self.assertEqual(coordinator.stats["recorded_forwards"], 1)
        self.assertEqual(coordinator.stats["model_tokens"], 1)
        self.assertEqual(coordinator.policy.heat[0], (0, 0, 1, 1))
        self.assertEqual(
            (coordinator.stats["route_hot"], coordinator.stats["route_total"]), (0, 4)
        )

    def test_unfinished_forward_is_startup_work_before_heat_and_error_after(self):
        coordinator = self.make_coordinator()
        x = torch.ones(1, 3, dtype=torch.bfloat16)
        weights, ids = torch.ones(1, 2), torch.tensor([[0, 1]])
        with self.forward_context(coordinator, torch.tensor([False])):
            for _ in range(2):
                for tier in coordinator.layers:
                    coordinator.begin_layer(tier, x, weights, ids)
                    coordinator.end_layer(tier)
            self.assertEqual(coordinator.stats["dropped_startup_records"], 1)
            coordinator.forward_rows = None
            coordinator.enable_heat()
            for tier in coordinator.layers:
                coordinator.begin_layer(tier, x, weights, ids)
                coordinator.end_layer(tier)
            # TierLayer.apply poisons the coordinator on this error.
            with self.assertRaises(RuntimeError):
                coordinator.begin_layer(coordinator.layers[0], x, weights, ids)

    def test_capture_time_forward_is_never_finished(self):
        coordinator = self.make_coordinator()
        x = torch.ones(1, 3, dtype=torch.bfloat16)
        weights = torch.ones(1, 2)
        with (
            self.forward_context(coordinator, torch.tensor([True])),
            patch.object(rt, "_is_capturing", return_value=True),
        ):
            # Capture must not wait on a previous stream: adopt it beforehand.
            coordinator.stream = SimpleNamespace(cuda_stream=3, synchronize=Mock())
            coordinator.stream_id = 3
            with self.assertRaises(RuntimeError):
                coordinator.begin_layer(
                    coordinator.layers[0], x, weights, torch.tensor([[-1, -1]])
                )
            coordinator.stream.synchronize.assert_not_called()
            coordinator.stream = coordinator.stream_id = None
            for tier in coordinator.layers:
                coordinator.begin_layer(tier, x, weights, torch.tensor([[-1, -1]]))
                coordinator.end_layer(tier)
            self.assertIsNone(coordinator.forward_rows)
            self.assertEqual(coordinator.stats["captured_forwards"], 1)
            with self.assertRaises(RuntimeError):
                coordinator.finish_forward(1)
        self.assertTrue(coordinator.poisoned)

    def test_finish_rejects_incomplete_sequence_shape_drift_and_overflow(self):
        coordinator = self.make_coordinator()
        coordinator.recorded = 1
        with self.assertRaises(RuntimeError):
            coordinator.finish_forward(1)
        coordinator.poisoned, coordinator.recorded = False, 0
        for rows in (0, 5):
            with self.assertRaises(ValueError):
                coordinator.finish_forward(rows)
            coordinator.poisoned = False
        x = torch.ones(1, 3, dtype=torch.bfloat16)
        weights, ids = torch.ones(1, 2), torch.tensor([[0, 1]])
        with self.forward_context(coordinator, torch.tensor([False])):
            coordinator.begin_layer(coordinator.layers[0], x, weights, ids)
            with self.assertRaises(ValueError):
                coordinator.begin_layer(
                    coordinator.layers[1],
                    x.expand(2, -1),
                    weights.expand(2, -1),
                    ids.expand(2, -1),
                )
            coordinator.recorded = 0
            with self.assertRaises(ValueError):
                coordinator.begin_layer(
                    coordinator.layers[0],
                    torch.ones(5, 3, dtype=torch.bfloat16),
                    torch.ones(5, 2),
                    torch.zeros(5, 2, dtype=torch.int64),
                )
            unallocated = rt.TierCoordinator(
                coordinator.layers, coordinator.settings, {}
            )
            with self.assertRaises(RuntimeError):
                unallocated.begin_layer(coordinator.layers[0], x, weights, ids)

    def test_waves_split_on_slot_reuse_and_temp_budget_only(self):
        from lab_expert_tier.tier_policy import Swap

        reuse = (Swap(0, 0, 0, 2, 0), Swap(0, 0, 1, 0, 1))
        self.assertEqual(rt.plan_waves(reuse, 8), [[reuse[0]], [reuse[1]]])
        spread = (Swap(0, 0, 0, 2, 0), Swap(1, 0, 0, 2, 0), Swap(0, 1, 1, 3, 1))
        self.assertEqual(rt.plan_waves(spread, 8), [list(spread)])
        self.assertEqual(rt.plan_waves(spread, 2), [list(spread[:2]), [spread[2]]])
        cold_reuse = (Swap(0, 0, 0, 2, 0), Swap(0, 1, 0, 3, 2))
        self.assertEqual(
            rt.plan_waves(cold_reuse, 8), [[cold_reuse[0]], [cold_reuse[1]]]
        )
        self.assertEqual(rt.plan_waves((), 8), [])
        with self.assertRaises(ValueError):
            rt.plan_waves(spread, 0)

    def test_wave_phases_wait_between_all_d2h_all_h2d_and_host_writes(self):
        events: list[Any] = []

        class Row:
            def __init__(self, tag):
                self.tag = tag

            def copy_(self, other, non_blocking=False):
                events.append((self.tag, other.tag))

        def bank(tag, rows):
            return {
                name: [Row(f"{tag}{i}") for i in range(rows)] for name in rt.TENSORS
            }

        temporary = {name: [Row("t0"), Row("t1")] for name in rt.TENSORS}
        items = [
            (bank("h", 2), bank("c", 2), rt.temporary_row(temporary, 0), 0, 1),
            (bank("H", 2), bank("C", 2), rt.temporary_row(temporary, 1), 1, 0),
        ]
        rt.swap_tensor_rows_wave(items, lambda: events.append("wait"))
        per_phase = len(rt.TENSORS)
        self.assertEqual(events[:per_phase], [("t0", "h0")] * per_phase)
        self.assertEqual(events[per_phase : 2 * per_phase], [("t1", "H1")] * per_phase)
        self.assertEqual(events[2 * per_phase], "wait")
        h2d = events[2 * per_phase + 1 : 4 * per_phase + 1]
        self.assertEqual(h2d, [("h0", "c1")] * per_phase + [("H1", "C0")] * per_phase)
        self.assertEqual(events[4 * per_phase + 1], "wait")
        host = events[4 * per_phase + 2 :]
        self.assertEqual(host, [("c1", "t0")] * per_phase + [("C0", "t1")] * per_phase)

    def test_wave_migration_matches_sequential_swaps_byte_for_byte(self):
        from lab_expert_tier.tier_policy import Swap

        def make_layer(index, seed):
            layer = object.__new__(rt.TierLayer)
            layer.index, layer.device = index, torch.device("cpu")
            layer.num_experts, layer.hot_slots, layer.cold_slots = 5, 2, 3
            layer.row_bytes = 6
            layer.hot = {
                name: (torch.arange(2 * 3) + 100 * seed + 10 * i)
                .reshape(2, 3)
                .to(torch.int32)
                for i, name in enumerate(rt.TENSORS)
            }
            layer.cold_cpu = {
                name: (torch.arange(3 * 3) + 100 * seed + 10 * i + 50)
                .reshape(3, 3)
                .to(torch.int32)
                for i, name in enumerate(rt.TENSORS)
            }
            layer.hot_map_host = (0, 1, -1, -1, -1)
            layer.cold_map_host = (-1, -1, 0, 1, 2)
            layer.hot_map = layer.cold_map = None
            layer.publish_maps()
            return layer

        # Plan order matters: the third swap reuses hot slot 0 of layer 0 and
        # must follow the first physically, so it opens a second wave; the
        # fourth reuses cold slot 0 from the first wave only and joins the
        # second. Layer 1's independent swap shares the first wave.
        plan = (
            Swap(0, 0, 0, 0, 2),
            Swap(1, 1, 2, 1, 4),
            Swap(0, 0, 1, 2, 3),
            Swap(0, 1, 0, 1, 0),
        )
        sequential = [make_layer(i, i + 1) for i in range(2)]
        temp = {name: torch.zeros(3, dtype=torch.int32) for name in rt.TENSORS}
        for swap in plan:
            sequential[swap.layer].swap(
                swap.old_expert, swap.new_expert, swap.hot_slot, swap.cold_slot, temp
            )
        waved = [make_layer(i, i + 1) for i in range(2)]
        pool = {name: torch.zeros(8, 3, dtype=torch.int32) for name in rt.TENSORS}
        coordinator = rt.TierCoordinator(waved, rt.Settings(32 * 2**30), pool)
        coordinator.device = torch.device("cpu")
        coordinator.migrate(plan)
        self.assertEqual(coordinator.stats["migration_waves"], 2)
        self.assertEqual(coordinator.stats["max_wave_swaps"], 2)
        self.assertEqual(coordinator.stats["swaps"], 4)
        self.assertEqual(coordinator.per_layer_swaps, [3, 1])
        for a, b in zip(sequential, waved):
            self.assertEqual(
                (a.hot_map_host, a.cold_map_host), (b.hot_map_host, b.cold_map_host)
            )
            self.assertEqual(a.hot_map.tolist(), b.hot_map.tolist())
            self.assertEqual(a.cold_map.tolist(), b.cold_map.tolist())
            for name in rt.TENSORS:
                self.assertTrue(torch.equal(a.hot[name], b.hot[name]), name)
                self.assertTrue(torch.equal(a.cold_cpu[name], b.cold_cpu[name]), name)
        self.assertEqual(waved[0].hot_map_host, (1, -1, -1, 0, -1))
        with self.assertRaises(AssertionError):
            coordinator.migrate((Swap(0, 0, 0, 0, 2),))

    def test_marlin_block_size_matches_stock_selection(self):
        def stock(tokens, top_k, local, global_, input_dtype):
            m = math.ceil(tokens * local / global_)
            for block in (8, 16, 32, 48, 64):
                if m * top_k / local / block < 0.9:
                    break
            if input_dtype is not None and input_dtype.itemsize == 1:
                block = max(block, 16)
            return block

        for tokens in (1, 2, 7, 64, 512):
            for top_k in (1, 4, 10):
                for local in (1, 3, 258, 254):
                    for dtype in (None, torch.int8, torch.float8_e4m3fn):
                        self.assertEqual(
                            rt.marlin_block_size(tokens, top_k, local, 512, dtype),
                            stock(tokens, top_k, local, 512, dtype),
                        )

    def test_fused_split_writes_disjoint_rows_once_and_zeros_padding(self):
        tier = object.__new__(rt.TierLayer)
        tier.settings = rt.Settings(32 * 2**30)
        tier.device = torch.device("cpu")
        tier.num_experts, tier.hot_slots, tier.cold_slots = 4, 2, 2
        tier.hot_map = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
        tier.cold_map = torch.tensor([-1, -1, 0, 1], dtype=torch.int32)
        tier.layer = SimpleNamespace(activation="silu")
        tier.marlin_workspace = torch.zeros(4, dtype=torch.int32)
        hidden, inner = 3, 5
        tier.hot = {"w13_weight": "hot13", "w2_weight": "hot2"}
        tier.cold = {"w13_weight": "cold13", "w2_weight": "cold2"}

        def experts(tag):
            return SimpleNamespace(
                w1_bias=None,
                w2_bias=None,
                w1_scale=f"{tag}-s1",
                w2_scale=f"{tag}-s2",
                quant_type_id=7,
                activation=f"{tag}-act",
                a1_gscale=None,
                a2_gscale=None,
                g1_alphas=f"{tag}-g1",
                g2_alphas=f"{tag}-g2",
                w13_g_idx=None,
                w2_g_idx=None,
                w13_g_idx_sort_indices=None,
                w2_g_idx_sort_indices=None,
                w1_zp=None,
                w2_zp=None,
                input_dtype=None,
                is_k_full=True,
                activation_config="cfg",
            )

        tier.hot_kernel = SimpleNamespace(fused_experts=experts("hot"))
        tier.cold_kernel = SimpleNamespace(fused_experts=experts("cold"))
        x = torch.ones(3, hidden, dtype=torch.bfloat16)
        ids = torch.tensor([[0, 2], [3, 1], [-1, -1]], dtype=torch.int32)
        weights = torch.tensor([[0.25, 0.75], [0.5, 0.5], [0.0, 0.0]])
        calls = []
        handed = []

        def get_simultaneous(*specs):
            # Garbage-filled workspace: only an explicit zero fill may be relied on.
            buffers = [torch.full(shape, 7, dtype=dtype) for shape, dtype in specs]
            handed.append(buffers)
            return buffers

        def align(topk_ids, block, num_experts, expert_map, ignore_invalid_experts):
            self.assertIs(topk_ids, ids)
            self.assertTrue(ignore_invalid_experts)
            self.assertEqual(num_experts, 4)
            return ("sorted", expert_map), "experts", "post"

        def fused(**kw):
            calls.append(kw)
            expert_map = kw["expert_map"]
            self.assertEqual(kw["sorted_token_ids"], ("sorted", expert_map))
            self.assertIs(kw["workspace"], tier.marlin_workspace)
            self.assertIs(kw["intermediate_cache13"], handed[-1][0])
            self.assertIs(kw["intermediate_cache2"], handed[-1][1])
            self.assertIs(kw["output"], handed[-1][2])
            self.assertFalse(kw["apply_router_weight_on_input"])
            for t in range(ids.shape[0]):
                for k in range(ids.shape[1]):
                    expert = int(ids[t, k])
                    if expert >= 0 and int(expert_map[expert]) >= 0:
                        kw["output"][t * 2 + k].fill_(
                            (expert + 1) * float(weights[t, k])
                        )
            return kw["output"]

        fused_moe = "vllm.model_executor.layers.fused_moe"
        modules = {
            fused_moe + ".experts.marlin_moe": SimpleNamespace(
                _fused_marlin_moe=fused,
                marlin_moe_intermediate_size=lambda w1, w2: inner,
            ),
            fused_moe + ".moe_align_block_size": SimpleNamespace(
                moe_align_block_size=align
            ),
            "vllm.scalar_type": SimpleNamespace(
                ScalarType=SimpleNamespace(from_id=lambda i: ("scalar", i))
            ),
            "vllm.v1.worker.workspace": SimpleNamespace(
                current_workspace_manager=lambda: SimpleNamespace(
                    get_simultaneous=get_simultaneous
                )
            ),
        }
        with patch.dict(sys.modules, modules):
            out = tier.split(x, weights, ids)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["w1"], "hot13")
        self.assertEqual(calls[1]["w1"], "cold13")
        self.assertIs(calls[0]["expert_map"], tier.hot_map)
        self.assertIs(calls[1]["expert_map"], tier.cold_map)
        self.assertEqual(calls[0]["w1_scale"], "hot-s1")
        self.assertEqual(calls[1]["global_scale2"], "cold-g2")
        self.assertEqual(calls[0]["quant_type"], ("scalar", 7))
        self.assertEqual(calls[0]["activation_func"], "hot-act")
        self.assertEqual(handed[-1][2].shape, (6, hidden))
        self.assertEqual(handed[-1][0].shape, (6 * max(2 * inner, hidden),))
        # Token 0: expert 0 (hot) * .25 + expert 2 (cold) * .75; token 1:
        # expert 3 (cold) * .5 + expert 1 (hot) * .5; token 2 is padding.
        expected = torch.tensor([[1 * 0.25 + 3 * 0.75], [4 * 0.5 + 2 * 0.5], [0.0]])
        torch.testing.assert_close(
            out.float(), expected.expand(3, hidden).float(), rtol=1e-2, atol=1e-2
        )
        self.assertTrue(torch.equal(out[2], torch.zeros_like(out[2])))

    def test_record_layer_receives_hot_map_and_report_marks_unmeasured_hits(self):
        coordinator = self.make_coordinator()
        received: list[Any] = []
        original = coordinator.observer.record_layer

        def record_layer(*args, **kwargs):
            received.append(kwargs.get("hot_map"))
            return original(*args, **kwargs)

        coordinator.observer.record_layer = record_layer
        tier = coordinator.layers[0]
        tier.hot_map = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
        x = torch.ones(1, 3, dtype=torch.bfloat16)
        with self.forward_context(coordinator, torch.tensor([False])):
            coordinator.begin_layer(tier, x, torch.ones(1, 2), torch.tensor([[0, 2]]))
        self.assertIs(received[0], tier.hot_map)
        with patch.object(rt.LOGGER, "warning") as log:
            coordinator.report()
        snapshot = json.loads(log.call_args.args[1])
        self.assertEqual(
            (snapshot["route_hot"], snapshot["route_hot_available"]), (0, True)
        )
        coordinator.observer = SimpleNamespace(
            capacity=4, records=None, records_host=None
        )
        with patch.object(rt.LOGGER, "warning") as log:
            coordinator.report()
        snapshot = json.loads(log.call_args.args[1])
        self.assertEqual(
            (snapshot["route_hot"], snapshot["route_hot_available"]), (None, False)
        )

    def test_device_observer_end_to_end_on_cpu(self):
        """The real device observer drives the real policy through the seam."""
        from lab_expert_tier import heat_device

        rt.OBSERVERS["device"] = heat_device.DeviceObserver
        try:
            observer = rt.make_observer(
                "device", num_layers=2, num_experts=4, decay=1.0, sync_period=1
            )
        finally:
            rt.OBSERVERS["device"] = "vllm._lab_expert_tier.heat_device:DeviceObserver"
        settings = rt.Settings(
            32 * 2**30, sync_tokens=1, swaps_per_token=2, decay=1.0, hysteresis=0
        )
        coordinator = self.make_coordinator(settings=settings)
        coordinator.observer = observer
        coordinator.allocate_records(torch.device("cpu"), 2, 4)
        for layer in coordinator.layers:
            layer.hot_map = torch.tensor(layer.hot_map_host, dtype=torch.int32)
            layer.hot, layer.cold_cpu, layer.row_bytes = {}, {}, 1
            layer.publish_maps = Mock()

            def stage_swap(old, new, hs, cs, layer=layer):
                layer.hot_map_host, layer.cold_map_host = rt.maps_after_swap(
                    layer.hot_map_host, layer.cold_map_host, old, new, hs, cs
                )

            layer.stage_swap = stage_swap
        x = torch.ones(1, 3, dtype=torch.bfloat16)
        weights = torch.ones(1, 2)

        def forward(ids):
            with self.forward_context(coordinator, torch.tensor([False])):
                for tier in coordinator.layers:
                    coordinator.begin_layer(tier, x, weights, ids)
                    coordinator.end_layer(tier)
            coordinator.finish_forward(1, 1)

        # Startup: nothing reaches the policy or the device heat. A dummy
        # forward reports a positive runner token count over an all-padding
        # mask (what warmup and capture look like); that mismatch must not
        # poison the first real snapshot after heat is enabled.
        forward(torch.tensor([[2, 3]]))
        with self.forward_context(coordinator, torch.tensor([True])):
            for tier in coordinator.layers:
                coordinator.begin_layer(tier, x, weights, torch.tensor([[-1, -1]]))
                coordinator.end_layer(tier)
        coordinator.finish_forward(1, 1)
        self.assertEqual(coordinator.policy.tokens_total, 0)
        self.assertEqual(coordinator.stats["ignored_startup_forwards"], 2)
        coordinator.enable_heat()
        # Cold experts 2 and 3 are selected every step; with sync_period 1 and
        # no hysteresis the policy must plan them into the hot slots.
        with patch.object(rt, "swap_tensor_rows_wave", lambda items, sync: None):
            for _ in range(3):
                forward(torch.tensor([[2, 3]]))
        self.assertGreater(coordinator.policy.tokens_total, 0)
        self.assertGreater(coordinator.stats["device_snapshots"], 0)
        self.assertGreater(coordinator.stats["swaps"], 0)
        self.assertEqual(coordinator.layers[0].hot_map_host[2:], (0, 1))
        self.assertFalse(coordinator.poisoned)
        with patch.object(rt.LOGGER, "warning") as log:
            coordinator.report()
        snapshot = json.loads(log.call_args.args[1])
        self.assertTrue(snapshot["route_hot_available"])

    def make_async_layer(self, index=0, hot=2, cold=3, spare=2, width=3):
        from lab_expert_tier.async_migration import SpareRing

        layer = object.__new__(rt.TierLayer)
        layer.index, layer.device = index, torch.device("cpu")
        layer.num_experts, layer.hot_slots, layer.cold_slots = hot + cold, hot, cold
        layer.staging_slots, layer.spare_slots = 0, spare
        layer.bank_rows, layer.cold_rows_total = hot + spare, cold + spare
        layer.row_bytes = 6
        layer.bank = {
            name: (torch.arange(hot * width) + 1000 * i)
            .reshape(hot, width)
            .to(torch.int32)
            for i, name in enumerate(rt.TENSORS)
        }
        layer.cold_cpu = {
            name: (torch.arange(cold * width) + 1000 * i + 500)
            .reshape(cold, width)
            .to(torch.int32)
            for i, name in enumerate(rt.TENSORS)
        }
        for name in rt.TENSORS:
            layer.bank[name] = torch.cat(
                (layer.bank[name], torch.full((spare, width), -1, dtype=torch.int32))
            )
            layer.cold_cpu[name] = torch.cat(
                (
                    layer.cold_cpu[name],
                    torch.full((spare, width), -1, dtype=torch.int32),
                )
            )
        layer.cold = layer.cold_cpu
        layer.hot = {name: t[:hot] for name, t in layer.bank.items()}
        layer.hot_rows, layer.cold_rows = list(range(hot)), list(range(cold))
        layer.vram_spares = SpareRing(range(hot, hot + spare))
        layer.ram_spares = SpareRing(range(cold, cold + spare))
        layer.hot_map_host = tuple(range(hot)) + (-1,) * cold
        layer.cold_map_host = (-1,) * hot + tuple(range(cold))
        layer.hot_map = layer.cold_map = None
        layer.publish_maps()
        return layer

    def test_async_enqueue_copies_into_spares_and_flip_retires_old_rows(self):
        from lab_expert_tier import async_migration as am
        from lab_expert_tier.tier_policy import Swap

        layer = self.make_async_layer()
        source_bank = {name: t.clone() for name, t in layer.bank.items()}
        source_cold = {name: t.clone() for name, t in layer.cold_cpu.items()}
        stream = am._migration_stream(layer.device)
        swap = Swap(0, 1, 2, 1, 4)  # hot slot 1 (expert 1) <-> cold slot 2 (expert 4)
        vram_spare, ram_spare = layer.enqueue_swap(swap, stream)
        self.assertEqual((vram_spare.row, ram_spare.row), (2, 3))
        for name in rt.TENSORS:
            # Promoted expert 4 (cold row 2) landed in VRAM row 2; evicted
            # expert 1 (hot row 1) landed in RAM row 3; nothing else moved.
            self.assertTrue(torch.equal(layer.bank[name][2], source_cold[name][2]))
            self.assertTrue(torch.equal(layer.cold_cpu[name][3], source_bank[name][1]))
            self.assertTrue(torch.equal(layer.bank[name][:2], source_bank[name][:2]))
            self.assertTrue(
                torch.equal(layer.cold_cpu[name][:3], source_cold[name][:3])
            )
        # Maps are untouched until the flip: the old placement stays in force.
        self.assertEqual(layer.hot_map.tolist(), [0, 1, -1, -1, -1])
        self.assertEqual(layer.hot_map_host, (0, 1, -1, -1, -1))
        layer.flip_swap(swap, vram_spare, ram_spare, "retire-fence")
        layer.publish_maps()
        self.assertEqual(layer.hot_rows, [0, 2])
        self.assertEqual(layer.cold_rows, [0, 1, 3])
        self.assertEqual(layer.hot_map_host, (0, -1, -1, -1, 1))
        self.assertEqual(layer.cold_map_host, (-1, 2, 0, 1, -1))
        # Device maps resolve logical slots to the physical rows.
        self.assertEqual(layer.hot_map.tolist(), [0, -1, -1, -1, 2])
        self.assertEqual(layer.cold_map.tolist(), [-1, 3, 0, 1, -1])
        retired_vram, retired_ram = layer.vram_spares.pop(), layer.ram_spares.pop()
        self.assertEqual((retired_vram.row, retired_vram.fence), (3, None))
        self.assertEqual((retired_ram.row, retired_ram.fence), (4, None))
        retired_vram, retired_ram = layer.vram_spares.pop(), layer.ram_spares.pop()
        self.assertEqual((retired_vram.row, retired_vram.fence), (1, "retire-fence"))
        self.assertEqual((retired_ram.row, retired_ram.fence), (2, "retire-fence"))
        # The synchronous path resolves the same physical rows.
        self.assertEqual((layer.resolve_hot_row(1), layer.resolve_cold_row(2)), (2, 3))

    def test_async_plan_commits_at_next_boundary_before_observation(self):
        from lab_expert_tier import async_migration as am

        settings = rt.Settings(
            32 * 2**30,
            sync_tokens=1,
            swaps_per_token=2,
            decay=1,
            hysteresis=1,
            temp_slots=2,
            async_migration=True,
        )
        coordinator = self.make_coordinator(settings=settings)
        for i, spec in enumerate(coordinator.layers):
            layer = self.make_async_layer(index=i, hot=2, cold=2, spare=2)
            layer.publish_maps = Mock(wraps=layer.publish_maps)
            coordinator.layers[i] = layer
        events: list[Any] = []

        class PendingEvent:
            def query(self):
                return False

            def synchronize(self):
                events.append("sync-wait")

        pending_event = PendingEvent()
        order: list[str] = []

        def record_event(stream):
            order.append("record")
            return pending_event

        def wait_event(stream, event):
            order.append("wait")

        original_finish = coordinator.observer.finish

        def finish(*args, **kwargs):
            order.append("observe")
            return original_finish(*args, **kwargs)

        coordinator.observer.finish = finish
        coordinator.observer.rebase = lambda **state: order.append("rebase")
        coordinator.enable_heat()
        record = torch.tensor([[2, 2, 1, 1, 1]], dtype=torch.int32)
        with (
            patch.object(am, "_record_event", record_event),
            patch.object(am, "_stream_wait_event", wait_event),
        ):
            # Boundary N: the plan is enqueued, not committed.
            self.replay(coordinator, record)
            self.assertIsNotNone(coordinator.pending)
            self.assertEqual(coordinator.stats["async_plans"], 1)
            self.assertEqual(coordinator.stats["swaps"], 0)
            self.assertEqual(coordinator.policy.version, 0)
            self.assertEqual(coordinator.layers[0].hot_map_host, (0, 1, -1, -1))
            for layer in coordinator.layers:
                layer.publish_maps.assert_not_called()
            # Boundary N+1: the transfer is still running, so the coordinator
            # waits, flips, commits, rebases, and only then observes. This
            # forward selects only hot experts, so no new plan follows.
            order.clear()
            hot_only = torch.tensor([[2, 1, 1, 1, 1]], dtype=torch.int32)
            self.replay(coordinator, hot_only)
        self.assertEqual(events, ["sync-wait"])
        self.assertEqual(order[:4], ["wait", "record", "rebase", "observe"])
        self.assertIsNone(coordinator.pending)
        self.assertEqual(coordinator.stats["async_commits"], 1)
        self.assertEqual(coordinator.stats["swaps"], 2)
        self.assertEqual(coordinator.policy.version, 1)
        for layer in coordinator.layers:
            self.assertEqual(layer.hot_map_host[2], 0)
            layer.publish_maps.assert_called_once_with()
        self.assertEqual(coordinator.stats["async_wait_seconds"] >= 0, True)

    def test_async_ineligible_plans_fall_back_to_the_synchronous_path(self):
        settings = rt.Settings(
            32 * 2**30,
            sync_tokens=1,
            swaps_per_token=4,
            decay=1,
            hysteresis=1,
            temp_slots=1,
            async_migration=True,
        )
        coordinator = self.make_coordinator(settings=settings)
        for i in range(len(coordinator.layers)):
            coordinator.layers[i] = self.make_async_layer(
                index=i, hot=2, cold=2, spare=1
            )
        coordinator.temporary = {
            name: torch.zeros(1, 3, dtype=torch.int32) for name in rt.TENSORS
        }
        coordinator.enable_heat()
        # Both cold experts become hot: two swaps per layer exceed one spare.
        record = torch.tensor([[2, 3, 1, 1, 1]], dtype=torch.int32)
        self.replay(coordinator, record)
        self.assertIsNone(coordinator.pending)
        self.assertEqual(coordinator.stats["sync_fallbacks"], 1)
        self.assertEqual(coordinator.stats["fallback_over_budget"], 1)
        self.assertEqual(coordinator.stats["swaps"], 4)
        self.assertEqual(coordinator.policy.version, 1)
        self.assertEqual(coordinator.layers[0].hot_map_host, (-1, -1, 0, 1))
        self.assertEqual(coordinator.layers[0].hot_map.tolist(), [-1, -1, 0, 1])

    def test_flush_settles_a_pending_transaction_before_importing(self):
        from lab_expert_tier import async_migration as am

        settings = rt.Settings(
            32 * 2**30,
            sync_tokens=1,
            swaps_per_token=2,
            decay=1,
            hysteresis=1,
            temp_slots=2,
            async_migration=True,
        )
        coordinator = self.make_coordinator(settings=settings)
        for i in range(len(coordinator.layers)):
            coordinator.layers[i] = self.make_async_layer(
                index=i, hot=2, cold=2, spare=2
            )
        coordinator.enable_heat()
        record = torch.tensor([[2, 2, 1, 1, 1]], dtype=torch.int32)
        self.replay(coordinator, record)
        self.assertIsNotNone(coordinator.pending)
        coordinator.flush()
        self.assertIsNone(coordinator.pending)
        self.assertEqual(coordinator.stats["async_commits"], 1)
        with patch.object(rt.LOGGER, "warning") as log:
            coordinator.report()
        self.assertIn('"async_pending": false', log.call_args.args[1])
        with patch.dict(
            os.environ,
            {rt.PREFIX + "GIB": "32", rt.PREFIX + "ASYNC_MIGRATION": "1"},
            clear=True,
        ):
            self.assertTrue(rt.Settings.from_env().async_migration)
        env = {rt.PREFIX + "GIB": "32", rt.PREFIX + "ASYNC_MIGRATION": "2"}
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
            rt.Settings.from_env()
        del am

    def test_observer_registry_builds_default_and_rejects_unknown(self):
        observer = rt.make_observer(
            "records", num_layers=2, num_experts=4, decay=0.5, sync_period=1
        )
        self.assertIsInstance(observer, rt.RecordObserver)
        with self.assertRaises(ValueError):
            rt.make_observer("missing")
        with (
            patch.dict(
                os.environ,
                {rt.PREFIX + "GIB": "32", rt.PREFIX + "OBSERVER": "x y"},
                clear=True,
            ),
            self.assertRaises(ValueError),
        ):
            rt.Settings.from_env()
        with patch.dict(os.environ, {rt.PREFIX + "GIB": "32"}, clear=True):
            self.assertEqual(rt.Settings.from_env().observer, "records")

    def test_split_routes_batch_one_through_staging_and_one_chain(self):
        from lab_expert_tier import staging as st

        tier = object.__new__(rt.TierLayer)
        tier.settings = rt.Settings(32 * 2**30)
        tier.device = torch.device("cpu")
        tier.num_experts, tier.hot_slots, tier.cold_slots = 4, 2, 2
        tier.staging_slots = 2
        tier.hot_map = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
        tier.cold_map = torch.tensor([-1, -1, 0, 1], dtype=torch.int32)
        tier.bank = {"w13_weight": "bank13", "w2_weight": "bank2"}
        tier.hot = {"w13_weight": "hot13", "w2_weight": "hot2"}
        tier.cold = {"w13_weight": "cold13", "w2_weight": "cold2"}
        tier.staging = {"w13_weight": "stage13", "w2_weight": "stage2"}
        tier.bank_kernel = SimpleNamespace(fused_experts="bank-experts")
        tier.hot_kernel = SimpleNamespace(fused_experts="hot-experts")
        tier.cold_kernel = SimpleNamespace(fused_experts="cold-experts")
        calls: list[Any] = []
        planned = (torch.tensor([1, 0]), torch.tensor([0, 1, -1, 2]), torch.tensor(1))

        def plan(ids, cold_map, hot_map, hot_slots, staging_slots):
            calls.append(("plan", ids.tolist(), hot_slots, staging_slots))
            self.assertIs(cold_map, tier.cold_map)
            self.assertIs(hot_map, tier.hot_map)
            return planned

        def gather(source, staging, gather_index, count):
            calls.append(("gather", source, staging, gather_index.tolist()))

        def chains(x, weights, ids, partitions):
            calls.append(("chains", partitions))
            return "output"

        x = torch.ones(1, 3, dtype=torch.bfloat16)
        weights = torch.ones(1, 2)
        ids = torch.tensor([[3, 0]])
        with (
            patch.object(st, "plan_staging", plan),
            patch.object(st, "gather_staging", gather),
            patch.object(tier, "_run_marlin_chains", chains),
        ):
            self.assertEqual(tier.split(x, weights, ids), "output")
            # Two rows exceed the staging rows: the two-partition path runs.
            tier.split(x.expand(2, -1), weights.expand(2, -1), ids.expand(2, -1))
        self.assertEqual(calls[0], ("plan", [[3, 0]], 2, 2))
        self.assertEqual(calls[1], ("gather", tier.cold, tier.staging, [1, 0]))
        kind, partitions = calls[2]
        self.assertEqual(kind, "chains")
        self.assertEqual(len(partitions), 1)
        experts, tensors, expert_map, slots = partitions[0]
        self.assertEqual((experts, tensors, slots), ("bank-experts", tier.bank, 4))
        self.assertIs(expert_map, planned[1])
        kind, partitions = calls[3]
        self.assertEqual(len(partitions), 2)
        self.assertEqual(partitions[0][1], tier.hot)
        self.assertEqual(partitions[1][1], tier.cold)

    def test_runner_hook_is_noop_without_tier_and_forwards_padded_rows(self):
        rt.finish_model_forward(SimpleNamespace(), 8)
        coordinator = SimpleNamespace(finish_forward=Mock())
        rt.finish_model_forward(
            SimpleNamespace(_lab_expert_tier_coordinator=coordinator), 8
        )
        coordinator.finish_forward.assert_called_once_with(8, None)
        rt.finish_model_forward(
            SimpleNamespace(_lab_expert_tier_coordinator=coordinator), 8, 1
        )
        coordinator.finish_forward.assert_called_with(8, 1)

    def test_observer_seam_dispatches_legacy_deferred_and_device_snapshots(self):
        class Observer(rt.RecordObserver):
            def __init__(self):
                super().__init__()
                self.results: list[Any] = []
                self.calls: list[Any] = []
                self.gate_opened = 0
                self.acknowledged: list[Any] = []
                self.rebased: list[Any] = []

            def acknowledge_snapshot(self, snapshot):
                self.acknowledged.append(snapshot)

            def rebase(self, **state):
                self.rebased.append(state)

            def finish(self, rows, valid_rows, heat_enabled, stream, num_experts):
                self.calls.append((rows, valid_rows, heat_enabled))
                result = self.results.pop(0)
                if result == "legacy":
                    return super().finish(
                        rows, valid_rows, heat_enabled, stream, num_experts
                    )
                return result

            def flush(self):
                return self.results.pop(0) if self.results else None

            def on_heat_enabled(self):
                self.gate_opened += 1

        observer = Observer()
        coordinator = self.make_coordinator(
            settings=rt.Settings(32 * 2**30, sync_tokens=0)
        )
        coordinator.observer = observer
        coordinator.allocate_records(torch.device("cpu"), 2, 4)
        record = torch.tensor([[2, 3, 1, 1, 1]], dtype=torch.int32)
        coordinator.records[:1] = record[:, None, :]
        # Startup: a snapshot before heat is enabled is a contract violation.
        observer.results = [rt.DeviceSnapshot([[0.0] * 4] * 2, 1, 1, 0, 2)]
        with self.assertRaises(RuntimeError):
            coordinator.finish_forward(1, 1)
        coordinator.poisoned = False
        observer.results = [rt.Deferred()]
        coordinator.finish_forward(1, 1)
        self.assertEqual(coordinator.stats["ignored_startup_forwards"], 2)
        self.assertEqual(coordinator.stats["model_forwards"], 2)
        coordinator.enable_heat()
        self.assertEqual(observer.gate_opened, 1)
        # Legacy readback still observes and plans exactly as before.
        observer.results = ["legacy"]
        coordinator.finish_forward(1, 1)
        self.assertEqual(observer.calls[-1], (1, 1, True))
        self.assertEqual(coordinator.policy.tokens_total, 1)
        # Deferred forwards count but never plan against stale heat.
        observer.results = [rt.Deferred(), rt.Deferred(forwards=1)]
        coordinator.finish_forward(1, 1)
        coordinator.finish_forward(1, 1)
        self.assertEqual(coordinator.stats["deferred_forwards"], 2)
        self.assertEqual(coordinator.stats["model_forwards"], 3)
        self.assertEqual(coordinator.policy.tokens_total, 1)
        # A snapshot needs a policy importer; without one it fails closed.
        observer.results = [rt.DeviceSnapshot([[0.0] * 4] * 2, 2, 2, 1, 4)]
        coordinator.policy.import_snapshot = None
        with self.assertRaises(NotImplementedError):
            coordinator.finish_forward(1, 1)
        self.assertTrue(coordinator.poisoned)
        coordinator.poisoned = False
        imported: list[Any] = []

        def import_snapshot(snapshot):
            imported.append(snapshot)
            coordinator.policy.tokens_total = snapshot.tokens

        coordinator.policy.import_snapshot = import_snapshot
        # Cumulative totals: 3 tokens so far (1 legacy + 2 new), 2 forwards.
        observer.results = [rt.DeviceSnapshot([[0.0] * 4] * 2, 3, 2, 1, 4)]
        coordinator.finish_forward(1, 1)
        self.assertEqual(len(imported), 1)
        self.assertEqual(coordinator.stats["device_snapshots"], 1)
        self.assertEqual(coordinator.stats["model_tokens"], 3)
        self.assertEqual(
            (coordinator.stats["route_hot"], coordinator.stats["route_total"]), (1, 8)
        )
        # Forwards are counted once per finish; a snapshot window covering
        # two forwards is recorded separately and never re-added.
        self.assertEqual(coordinator.stats["model_forwards"], 5)
        self.assertEqual(coordinator.stats["snapshot_forwards"], 2)
        # flush collects a pending snapshot without counting a forward and,
        # by default, without planning.
        # A second cumulative snapshot adds only its increments.
        observer.results = [rt.DeviceSnapshot([[0.0] * 4] * 2, 4, 3, 1, 6)]
        with patch.object(coordinator, "_plan_and_migrate") as planner:
            coordinator.flush()
            planner.assert_not_called()
            observer.results = [rt.DeviceSnapshot([[0.0] * 4] * 2, 4, 3, 1, 6)]
            coordinator.flush(plan=True)
            planner.assert_called_once_with()
        self.assertEqual(len(imported), 3)
        self.assertEqual(coordinator.stats["model_forwards"], 5)
        self.assertEqual(coordinator.stats["model_tokens"], 4)
        self.assertEqual(coordinator.stats["snapshot_forwards"], 3)
        self.assertEqual(
            (coordinator.stats["route_hot"], coordinator.stats["route_total"]), (1, 10)
        )
        observer.results = [rt.DeviceSnapshot([[0.0] * 4] * 2, 4, 2, 1, 6)]
        with self.assertRaises(RuntimeError):
            coordinator.finish_forward(1, 1)
        coordinator.poisoned = False
        observer.results = [rt.DeviceSnapshot([[0.0] * 4] * 2, 1, 1, 0, 2, False)]
        with self.assertRaises(RuntimeError):
            coordinator.finish_forward(1, 1)
        coordinator.poisoned = False
        observer.results = ["unknown"]
        with self.assertRaises(TypeError):
            coordinator.finish_forward(1, 1)
        coordinator.poisoned = False
        with self.assertRaises(ValueError):
            coordinator.finish_forward(1, 2)
        coordinator.poisoned = False
        # Foreign result objects are recognized by shape: a device module's
        # own snapshot/deferred classes never import this module.
        foreign = SimpleNamespace(
            heat=[[0.0] * 4] * 2,
            tokens=5,
            forwards=1,
            route_total=2,
            error=False,
            session_id="other",
        )
        observer.results = [foreign, SimpleNamespace(forwards=3)]
        coordinator.finish_forward(1, 1)
        coordinator.finish_forward(1, 1)
        self.assertIs(observer.acknowledged[-1], foreign)
        self.assertEqual(
            observer.rebased[-1],
            {
                "tokens_total": coordinator.policy.tokens_total,
                "version": coordinator.policy.version,
                "last_sync_tokens": coordinator.policy.last_sync_tokens,
            },
        )
        self.assertEqual(len(observer.rebased), 4)
        self.assertEqual(coordinator.stats["deferred_forwards"], 5)
        observer.results = [SimpleNamespace(heat=[], tokens=0, forwards=1, error=True)]
        with self.assertRaises(RuntimeError):
            coordinator.finish_forward(1, 1)


if __name__ == "__main__":
    unittest.main()
