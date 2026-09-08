# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifecycle contracts, loading the real class without GPU import dependencies."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import torch


def load_state():
    source = Path(__file__).parents[3] / "vllm/models/qwen4_exp/nvidia/model_state.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )

    class Parent:
        def prepare_inputs(self, input_batch, req_states):
            return {}

    namespace: dict[str, Any] = {"Parent": Parent, "torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["Qwen4ExpModelState"]


def load_runner_ple_eligibility():
    source = Path(__file__).parents[3] / "vllm/v1/worker/gpu/model_runner.py"
    tree = ast.parse(source.read_text())
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner"
    )
    method = next(
        node
        for node in runner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_is_deferred_ple_eligible"
    )
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )

    class Modes:
        FULL = object()
        NONE = object()

    namespace: dict[str, Any] = {
        "CUDAGraphMode": Modes,
        "_PLE_DEFERRED_MAX_TOKENS": 8,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["_is_deferred_ple_eligible"], Modes


class DeferredStateTests(unittest.TestCase):
    def make_state(self, count=3):
        state = object.__new__(load_state())
        state._mmap_ple_modules = tuple(
            SimpleNamespace(deferred_rows=Mock()) for _ in range(count)
        )
        for module in state._mmap_ple_modules:
            module.deferred_rows.verify_consumed_rows.return_value = None
        state._deferred_ple_step = False
        state._deferred_ple_poisoned = False
        return state

    def test_complete_all_layers_and_next_step(self):
        state = self.make_state()
        state.set_deferred_ple_step(True)
        state.complete_deferred_ple()
        for module in state._mmap_ple_modules:
            module.deferred_rows.complete.assert_called_once()
            module.deferred_rows.abort.assert_not_called()
        self.assertFalse(state._deferred_ple_step)
        state.set_deferred_ple_step(True)
        self.assertTrue(state._deferred_ple_step)

    def test_verify_waits_until_all_fills_are_released(self):
        state = self.make_state()
        calls = []
        for i, module in enumerate(state._mmap_ple_modules):
            module.deferred_rows.complete.side_effect = lambda i=i: calls.append(
                ("fill", i)
            )
            module.deferred_rows.verify_consumed_rows.side_effect = lambda i=i: (
                calls.append(("verify", i))
            )
        state.set_deferred_ple_step(True)
        state.complete_deferred_ple()
        self.assertEqual(
            calls, [("fill", i) for i in range(3)] + [("verify", i) for i in range(3)]
        )

    def test_fill_failure_releases_unvisited_layers_and_poison_latches(self):
        state = self.make_state()
        state._mmap_ple_modules[0].deferred_rows.complete.side_effect = ValueError(
            "disk"
        )
        state.set_deferred_ple_step(True)
        with self.assertRaisesRegex(ValueError, "disk"):
            state.complete_deferred_ple()
        for module in state._mmap_ple_modules:
            module.deferred_rows.abort.assert_called_once()
            module.deferred_rows.verify_consumed_rows.assert_not_called()
        state._mmap_ple_modules[1].deferred_rows.complete.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            state.set_deferred_ple_step(False)

    def test_failed_release_still_attempts_every_layer(self):
        state = self.make_state()
        state.set_deferred_ple_step(True)
        state._mmap_ple_modules[0].deferred_rows.abort.side_effect = ValueError(
            "release"
        )
        with self.assertRaisesRegex(ValueError, "release"):
            state.abort_deferred_ple()
        for module in state._mmap_ple_modules:
            module.deferred_rows.abort.assert_called_once()
        self.assertTrue(state._deferred_ple_poisoned)

    def test_prepare_failure_releases_already_prepared_layers(self):
        state = self.make_state()
        state.uses_ngram_embedding = True
        state.ple_query_start_loc = torch.zeros(2, dtype=torch.int32)
        state._prepare_ngram_context = Mock(return_value=torch.zeros((1, 2)))
        batch = SimpleNamespace(
            num_reqs_after_padding=1,
            num_tokens=1,
            num_tokens_after_padding=1,
            num_reqs=1,
            input_ids=torch.tensor([3]),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        )
        for module in state._mmap_ple_modules:
            module.prepare_deferred_mmap_rows = Mock()
        state._mmap_ple_modules[1].prepare_deferred_mmap_rows.side_effect = ValueError(
            "ids"
        )
        state.set_deferred_ple_step(True)
        with self.assertRaisesRegex(ValueError, "ids"):
            state.prepare_inputs(batch, None)
        state._mmap_ple_modules[0].prepare_deferred_mmap_rows.assert_called_once()
        state._mmap_ple_modules[2].prepare_deferred_mmap_rows.assert_not_called()
        for module in state._mmap_ple_modules:
            module.deferred_rows.abort.assert_called_once()

    def test_ngram_context_stops_at_committed_prefix(self):
        state = self.make_state(0)
        state.ngram_context = torch.full((1, 2), 99, dtype=torch.int32)
        state.ngram_context_offsets = torch.tensor([-2, -1], dtype=torch.int64)
        state.ngram_eos_token_id = 99

        input_batch = SimpleNamespace(
            num_reqs=1,
            num_reqs_after_padding=1,
            idx_mapping=torch.tensor([0], dtype=torch.int32),
        )
        all_token_ids = torch.tensor(
            [[101, 102, 103, 104, 105, 900, 901]], dtype=torch.int32
        )
        req_states = SimpleNamespace(
            num_computed_tokens=SimpleNamespace(
                gpu=torch.tensor([0], dtype=torch.int32)
            ),
            all_token_ids=SimpleNamespace(gpu=all_token_ids),
        )

        expected = (
            torch.tensor([99, 99], dtype=torch.int32),
            torch.tensor([99, 101], dtype=torch.int32),
            torch.tensor([103, 104], dtype=torch.int32),
        )
        for committed, want in zip((0, 1, 4), expected):
            req_states.num_computed_tokens.gpu[0] = committed
            before = all_token_ids.clone()
            actual = state._prepare_ngram_context(input_batch, req_states)
            self.assertTrue(torch.equal(actual[0], want))
            self.assertTrue(torch.equal(all_token_ids, before))

        # During verification, the committed boundary advances by the number
        # of accepted tokens. Rejected candidate rows remain beyond that
        # boundary and therefore cannot become the next context by accident.
        for num_rejected, want in (
            (2, (102, 103)),
            (1, (103, 104)),
            (0, (104, 105)),
        ):
            committed = 2 + 3 - num_rejected
            req_states.num_computed_tokens.gpu[0] = committed
            before = all_token_ids.clone()
            actual = state._prepare_ngram_context(input_batch, req_states)
            self.assertEqual(actual[0].tolist(), list(want))
            self.assertTrue(torch.equal(all_token_ids, before))

    def test_prepare_deferred_uses_real_prefix_and_padded_width(self):
        state = self.make_state(2)
        state.uses_ngram_embedding = True
        state.ple_query_start_loc = torch.zeros(3, dtype=torch.int32)
        state._prepare_ngram_context = Mock(return_value=torch.zeros((2, 2)))
        batch = SimpleNamespace(
            num_reqs_after_padding=2,
            num_tokens=3,
            num_tokens_after_padding=8,
            num_reqs=2,
            input_ids=torch.tensor([101, 102, 103, 900, 901, 902, 903, 904]),
            query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        )
        for module in state._mmap_ple_modules:
            module.prepare_deferred_mmap_rows = Mock()

        state.set_deferred_ple_step(True)
        state.prepare_inputs(batch, None)

        for module in state._mmap_ple_modules:
            module.prepare_deferred_mmap_rows.assert_called_once()
            args = module.prepare_deferred_mmap_rows.call_args.args
            self.assertEqual(args[0].tolist(), [101, 102, 103])
            self.assertEqual(args[1].tolist(), [0, 1, 3])
            self.assertEqual(tuple(args[2].shape), (2, 2))
            self.assertEqual(args[3:], (3, 8))

    def test_runner_deferred_eligibility_covers_small_full_batches(self):
        eligible, modes = load_runner_ple_eligibility()

        def check(
            actual: int,
            padded: int,
            *,
            dummy: bool = False,
            mode: object = modes.FULL,
            expected: bool,
        ) -> None:
            batch = SimpleNamespace(
                num_tokens=actual,
                num_tokens_after_padding=padded,
                num_reqs=2,
            )
            descriptor = SimpleNamespace(cg_mode=mode)
            self.assertIs(eligible(batch, descriptor, dummy), expected)

        for width in (1, 2, 3, 8):
            check(width, width, expected=True)
        check(3, 8, expected=True)
        check(0, 1, expected=False)
        check(3, 2, expected=False)
        check(9, 9, expected=False)
        check(3, 8, dummy=True, expected=False)
        check(3, 8, mode=modes.NONE, expected=False)

    def test_disabled_and_empty_have_no_effect(self):
        for state in (self.make_state(), self.make_state(0)):
            state.set_deferred_ple_step(False)
            state.complete_deferred_ple()
            state.abort_deferred_ple()
            self.assertFalse(state._deferred_ple_poisoned)
        state = self.make_state()
        state._mmap_ple_modules[1].deferred_rows = None
        state.set_deferred_ple_step(True)
        self.assertFalse(state._deferred_ple_step)


if __name__ == "__main__":
    unittest.main()
