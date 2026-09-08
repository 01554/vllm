# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MRv2 n-gram speculator drafts from the bound request history and
fills unmatched lanes with a legal token (no device kernel needed here)."""

import importlib.util
import sys
import unittest
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

HERE = Path(__file__).resolve().parents[2] / "vllm" / "v1" / "worker" / "gpu"
spec = cast(
    ModuleSpec,
    importlib.util.spec_from_file_location(
        "ngram_speculator", HERE / "spec_decode" / "ngram" / "speculator.py"
    ),
)


def load_module():
    """Load the speculator source with its vLLM imports stubbed out."""
    stubs: dict[str, Any] = {
        "vllm.config": SimpleNamespace(VllmConfig=object),
        "vllm.config.compilation": SimpleNamespace(CUDAGraphMode=object),
        "vllm.v1.worker.gpu.input_batch": SimpleNamespace(InputBatch=object),
        "vllm.v1.worker.gpu.spec_decode.speculator": SimpleNamespace(
            BaseSpeculator=object
        ),
        "vllm.forward_context": SimpleNamespace(
            set_forward_context=lambda *a, **k: _Null()
        ),
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        module = importlib.util.module_from_spec(spec)
        cast(Loader, spec.loader).exec_module(module)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeKernel:
    """Returns -1 for ineligible rows and a fixed pattern otherwise."""

    def __init__(self, k):
        self.k = k
        self.calls: list[Any] = []

    def __call__(self, num_tokens, token_ids, eligible):
        self.calls.append((num_tokens.clone(), token_ids.clone(), eligible.clone()))
        rows = token_ids.shape[0]
        drafts = torch.full((rows, self.k), -1, dtype=torch.int32)
        for r in range(rows):
            if bool(eligible[r]):
                n = int(num_tokens[r])
                # Echo the last two tokens then leave the rest unmatched.
                drafts[r, 0] = token_ids[r, n - 2]
                drafts[r, 1] = token_ids[r, n - 1]
        return drafts, (drafts >= 0).int().sum(1)


class NgramSpeculatorTests(unittest.TestCase):
    def test_draft_from_history_masks_and_fills(self):
        module = load_module()
        kernel = FakeKernel(k=3)
        proposer = SimpleNamespace(kernel=kernel, vllm_config=None)
        token_ids = torch.tensor(
            [[5, 6, 7, 8, 0, 0], [9, 9, 9, 9, 9, 9], [1, 2, 3, 4, 5, 6]],
            dtype=torch.int32,
        )
        num_tokens = torch.tensor([4, 1, 6], dtype=torch.int32)
        eligible = torch.tensor([True, False, True])
        last_sampled = torch.tensor([[8], [9], [6], [42]], dtype=torch.int64)
        idx = torch.tensor([0, 1, 2])
        drafts = module.draft_from_history(
            proposer, token_ids, num_tokens, eligible, last_sampled, idx, 3
        )
        self.assertEqual(drafts.dtype, torch.int64)
        # Row 0: matched [7, 8] then filled with its last sampled token 8;
        # row 1 ineligible: all lanes filled with 9; row 2: [5, 6, 6].
        self.assertEqual(drafts.tolist(), [[7, 8, 8], [9, 9, 9], [5, 6, 6]])
        with self.assertRaises(RuntimeError):
            module.draft_from_history(
                proposer, token_ids, num_tokens, eligible, last_sampled, idx, 4
            )

    def test_propose_gathers_bound_state_by_batch_index(self):
        module = load_module()
        kernel = FakeKernel(k=2)
        spec = object.__new__(module.NgramSpeculator)
        spec.proposer = SimpleNamespace(kernel=kernel, vllm_config=None)
        spec.k, spec.min_n = 2, 3
        all_ids = torch.zeros(4, 8, dtype=torch.int32)
        all_ids[2, :5] = torch.tensor([11, 12, 13, 14, 15])
        all_ids[0, :2] = torch.tensor([21, 22])
        total_len = torch.tensor([2, 0, 5, 0], dtype=torch.int32)
        spec.bind_request_states(
            SimpleNamespace(
                all_token_ids=SimpleNamespace(gpu=all_ids),
                total_len=SimpleNamespace(gpu=total_len),
            )
        )
        batch = SimpleNamespace(num_reqs=2, idx_mapping=torch.tensor([2, 0, -1]))
        last_sampled = torch.tensor([[22], [0], [15], [0]], dtype=torch.int64)
        drafts = spec.propose(
            batch,
            {},
            {},
            None,
            None,
            num_sampled=torch.tensor([1, 1, 0], dtype=torch.int32),
            num_rejected=torch.zeros(3, dtype=torch.int32),
            last_sampled=last_sampled,
            next_prefill_tokens=None,
            temperature=None,
            seeds=None,
        )
        # Request 2 (5 tokens) is eligible: drafts [14, 15]; request 0 has
        # only 2 tokens (< min_n): both lanes filled with its last token 22.
        self.assertEqual(drafts.tolist(), [[14, 15], [22, 22]])
        num_tokens, token_ids, eligible = kernel.calls[-1]
        self.assertEqual(num_tokens.tolist(), [5, 2])
        self.assertEqual(eligible.tolist(), [True, False])
        self.assertEqual(token_ids[0, :5].tolist(), [11, 12, 13, 14, 15])

    def test_unbound_speculator_fails_closed(self):
        module = load_module()
        spec = object.__new__(module.NgramSpeculator)
        spec._all_token_ids = spec._total_len = None
        with self.assertRaises(RuntimeError):
            spec.propose(
                SimpleNamespace(num_reqs=1, idx_mapping=torch.tensor([0])),
                {},
                {},
                None,
                None,
                torch.ones(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, 1, dtype=torch.int64),
                None,
                None,
                None,
            )


if __name__ == "__main__":
    unittest.main()
