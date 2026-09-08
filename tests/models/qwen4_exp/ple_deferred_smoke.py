# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-only smoke for the opt-in PLE WAIT path; run with a process timeout.

Build vllm_ple_wait_ext against this Python's torch first. This script checks
actual stream WAIT/capture and row bytes, not full-model runner integration.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def load_helper():
    path = Path(__file__).parents[3] / "vllm/models/qwen4_exp/nvidia/ple_wait.py"
    spec = importlib.util.spec_from_file_location("ple_wait_smoke_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DeferredRows


class Table:
    def __init__(self, dtype):
        self.torch_dtype = dtype
        # Raw encoded bytes cover dtype reinterpretation, including BF16.
        values = torch.arange(64 * 16, dtype=torch.float32).reshape(64, 16) / 16
        self.values = values.to(dtype)
        self.raw = self.values.view(torch.uint8).numpy()
        self.fail = False

    def gather(self, ids):
        if self.fail:
            raise OSError("injected table error")
        return self.raw[ids.reshape(-1)].copy()


def run(dtype):
    helper_cls = load_helper()
    tables = [Table(dtype) for _ in range(3)]
    destinations = [torch.zeros((1, 4, 16), dtype=dtype, device="cuda") for _ in tables]
    helpers = [helper_cls(dst, table) for dst, table in zip(destinations, tables)]
    ids = [torch.zeros((1, 4), dtype=torch.int64, device="cuda") for _ in tables]
    output = [torch.empty_like(dst) for dst in destinations]
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for helper in helpers:
            helper.prepare_dummy()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            for helper, dst, out in zip(helpers, destinations, output):
                helper.consume()
                out.copy_(dst)
    torch.cuda.current_stream().wait_stream(capture_stream)
    for step in range(5):
        expected = []
        for index, (helper, table, gpu_ids) in enumerate(zip(helpers, tables, ids)):
            values = np.array([[step + index, 3, 3, 63 - step]], dtype=np.int64)
            gpu_ids.copy_(torch.from_numpy(values))
            helper.prepare(gpu_ids)
            expected.append(table.values[values.reshape(-1)].reshape(1, 4, 16))
        graph.replay()
        for helper in helpers:
            helper.complete()
        torch.accelerator.synchronize()
        for actual, want in zip(output, expected):
            assert torch.equal(actual.cpu().view(torch.uint8), want.view(torch.uint8))
    # Release all layers on a host read failure, including those never filled.
    tables[0].fail = True
    for helper, gpu_ids in zip(helpers, ids):
        helper.prepare(gpu_ids)
    graph.replay()
    try:
        helpers[0].complete()
    except OSError:
        for helper in helpers:
            helper.abort()
    else:
        raise AssertionError("injected failure was not raised")
    torch.accelerator.synchronize()
    assert all(helper.poisoned for helper in helpers)
    print(f"PASS dtype={dtype}: 3 layers, 5 replays, duplicate IDs, raw bytes, abort")


if __name__ == "__main__":
    run(torch.bfloat16)
    run(torch.float8_e4m3fn)
