# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real memop/copy capture smoke; checkpoint integration is a separate gate."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.models.qwen4_exp.nvidia import ple_layer
from vllm.models.qwen4_exp.nvidia.ple_wait import (
    DeferredRows,
    StreamMemopsUnavailable,
)


class _Table:
    def __init__(self, source):
        self.source = source

    def gather(self, ids):
        # Production mmap gather consumes a flattened ID array.
        assert ids.ndim == 1
        return np.stack(
            [self.source[int(ids[h]), h].view(torch.uint8).numpy() for h in range(2)]
        )


def test_complete_passes_flat_ids_to_cuda_smoke_table():
    source = torch.arange(64, dtype=torch.bfloat16).reshape(8, 2, 4)
    helper = object.__new__(DeferredRows)
    helper.table = _Table(source)
    helper.ids = torch.tensor([[5, 3]], dtype=torch.int64)
    helper.rows = torch.empty((1, 2, 4), dtype=torch.bfloat16)
    helper._poisoned = False
    helper._pending = True
    helper._readback_event = Mock()
    helper._ext = Mock()
    helper.flag = torch.zeros(1, dtype=torch.int64)

    helper.complete()

    expected = torch.stack([source[5, 0], source[3, 1]])[None]
    assert torch.equal(helper.rows.view(torch.uint8), expected.view(torch.uint8))
    helper._readback_event.synchronize.assert_called_once_with()
    helper._ext.signal_flag.assert_called_once_with(helper.flag.data_ptr())
    assert not helper.pending


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_runtime_none_capture_replays_fresh_rows_and_resets_flag():
    # Distinct raw BF16 rows catch stale data, zeros and ID mixups without
    # relying on floating point comparisons or a model's generated text.
    source = torch.arange(64, dtype=torch.bfloat16).reshape(8, 2, 4)

    destination = torch.empty((1, 2, 4), dtype=torch.bfloat16, device="cuda")
    try:
        helper = DeferredRows(destination, _Table(source))
    except StreamMemopsUnavailable as exc:
        pytest.skip(str(exc))
    helper.prepare_dummy()
    graph = torch.cuda.CUDAGraph()
    context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        no_compile_layers={
            "ple": SimpleNamespace(ple_embedding=SimpleNamespace(deferred_rows=helper))
        },
    )
    with (
        patch.object(ple_layer, "get_forward_context", return_value=context),
        torch.cuda.graph(graph),
    ):
        ple_layer.qwen4_exp_ple_deferred_rows(destination, "ple")

    for row_ids in ([1, 2], [5, 3], [0, 7]):
        ids = torch.tensor([row_ids], dtype=torch.int64, device="cuda")
        helper.prepare(ids)
        graph.replay()
        # Never synchronize a replay waiting for the host before releasing it.
        try:
            helper.complete()
        except BaseException:
            helper.abort()
            raise
        torch.accelerator.synchronize()
        expected = torch.stack([source[row_ids[h], h] for h in range(2)])[None]
        assert torch.equal(
            destination.cpu().view(torch.uint8), expected.view(torch.uint8)
        )
        assert int(helper.flag[0]) == 0
        assert not helper.pending
