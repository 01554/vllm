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

    namespace: dict[str, Any] = {"Parent": Parent}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["Qwen4ExpModelState"]


class DeferredStateTests(unittest.TestCase):
    def make_state(self, count=3):
        state = object.__new__(load_state())
        state._mmap_ple_modules = tuple(
            SimpleNamespace(deferred_rows=Mock()) for _ in range(count)
        )
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
