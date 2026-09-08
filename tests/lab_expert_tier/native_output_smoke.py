# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The adapter's output alias is consumed by the runner's out-of-place add
before the next gemv rewrites the workspace: real adapter, real call order
(gemv -> shared + out -> next gemv), eager and captured graph replay.

    python tests/lab_expert_tier/native_output_smoke.py [--cpu-only]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# isort: off
from test_native_nvfp4 import make_bank  # noqa: E402  (installs the lab package)
from lab_expert_tier import native_nvfp4 as native  # noqa: E402
# isort: on


def step(bank, workspace, x, weights, ids, mapping, shared):
    out = native.gemv(x, weights, ids, bank, mapping, workspace)  # alias
    return shared + out  # the MoE runner's out-of-place combine


def run(cpu_only):
    device = "cpu" if cpu_only else "cuda"
    bank = {k: v.to(device) for k, v in make_bank().items()}
    workspace = native.allocate_workspace(bank, 2, 4, num_experts=3)
    x = torch.full((2, 32), 0.0625, dtype=torch.bfloat16, device=device)
    weights = torch.tensor([[0.25, 0.5, 0.125, 99.0]] * 2, device=device)
    ids_a = torch.tensor([[0, 2, 0, -1]] * 2, dtype=torch.int32, device=device)
    ids_b = torch.tensor([[1, 1, -1, -1]] * 2, dtype=torch.int32, device=device)
    mapping = torch.tensor([2, 0, 1, -1], dtype=torch.int32, device=device)
    shared = torch.full((2, 32), 1.0, dtype=torch.bfloat16, device=device)
    # Reference values from cloned outputs.
    ref_a = shared + native.gemv(x, weights, ids_a, bank, mapping, workspace).clone()
    ref_b = shared + native.gemv(x, weights, ids_b, bank, mapping, workspace).clone()
    assert not torch.equal(ref_a, ref_b), "routes must produce different outputs"
    first = step(bank, workspace, x, weights, ids_a, mapping, shared)
    second = step(bank, workspace, x, weights, ids_b, mapping, shared)
    assert torch.equal(first, ref_a) and torch.equal(second, ref_b), "eager alias"
    print("PASS eager: alias consumed before the next gemv")
    if cpu_only:
        return
    ids_buf = ids_a.clone()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            step(bank, workspace, x, weights, ids_buf, mapping, shared)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        first = step(bank, workspace, x, weights, ids_buf, mapping, shared)
        second = step(bank, workspace, x, weights, ids_b, mapping, shared)
    for replay in range(3):
        ids_buf.copy_(ids_a if replay % 2 == 0 else ids_b)
        graph.replay()
        torch.accelerator.synchronize()
        expected_first = ref_a if replay % 2 == 0 else ref_b
        assert torch.equal(first, expected_first), ("graph first", replay)
        assert torch.equal(second, ref_b), ("graph second", replay)
        print(f"PASS graph replay {replay}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    run(parser.parse_args().cpu_only)
