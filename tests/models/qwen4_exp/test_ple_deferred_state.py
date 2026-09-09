# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifecycle contracts for the model state."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm.config import CUDAGraphMode
from vllm.models.qwen4_exp.nvidia import ple_layer
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


class DeferredStateTests(unittest.TestCase):
    def test_consume_records_during_full_capture_runtime_none(self):
        for mode, capturing, rows, expected in (
            (CUDAGraphMode.NONE, True, 1, True),
            (CUDAGraphMode.NONE, False, 1, False),
            (CUDAGraphMode.PIECEWISE, True, 1, False),
            (CUDAGraphMode.FULL, False, 1, True),
            (CUDAGraphMode.NONE, True, 2, False),
        ):
            with self.subTest(mode=mode, capturing=capturing, rows=rows):
                helper = Mock()
                context = SimpleNamespace(
                    cudagraph_runtime_mode=mode,
                    no_compile_layers={
                        "ple": SimpleNamespace(
                            ple_embedding=SimpleNamespace(deferred_rows=helper)
                        )
                    },
                )
                output = torch.empty(rows, 2, 3)
                with (
                    patch.object(
                        ple_layer, "get_forward_context", return_value=context
                    ),
                    patch.object(
                        torch.cuda,
                        "is_current_stream_capturing",
                        return_value=capturing,
                    ),
                ):
                    ple_layer.qwen4_exp_ple_deferred_rows(output, "ple")
                if expected:
                    helper.consume.assert_called_once_with(destination=output)
                else:
                    helper.consume.assert_not_called()

    def make_state(self, count=3):
        state = object.__new__(Qwen4ExpModelState)
        state._mmap_ple_modules = tuple(
            SimpleNamespace(deferred_rows=Mock()) for _ in range(count)
        )
        state._deferred_ple_step = False
        state._deferred_ple_poisoned = False
        return state

    def test_mixed_capability_disables_all_helpers_before_capture(self):
        state = self.make_state()
        state.device = torch.device("cpu")
        state.max_num_tokens = 8
        modules = state._mmap_ple_modules
        modules[1].deferred_rows = None
        for module in modules:
            module.mmap_staging_nbytes = Mock(return_value=16)
            module.initialize_mmap_staging = Mock()
        with patch(
            "vllm.models.qwen4_exp.nvidia.model_state.MemorySnapshot",
            return_value=SimpleNamespace(free_memory=1024),
        ):
            state._initialize_mmap_staging(modules)
        self.assertTrue(all(m.deferred_rows is None for m in modules))
        for module in modules:
            module.initialize_mmap_staging.assert_called_once_with(8, state.device)
        state.set_deferred_ple_step(True)
        self.assertFalse(state._deferred_ple_step)

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
        with (
            patch.object(MambaHybridModelState, "prepare_inputs", return_value={}),
            self.assertRaisesRegex(ValueError, "ids"),
        ):
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
