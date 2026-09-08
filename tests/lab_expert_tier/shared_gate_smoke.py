# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU smoke for the fused shared-expert gate: the Triton program must match
vLLM's dot -> sigmoid -> multiply sequence on CUDA (eager and graph replay).

    python tests/lab_expert_tier/shared_gate_smoke.py [--cpu-only]
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_shared_gate  # noqa: E402,F401  (installs the lab package)
from lab_expert_tier import shared_gate as sg  # noqa: E402


def torch_path(x, w, out):
    return F.sigmoid(F.linear(x, w)) * out


def run(cpu_only):
    device = "cpu" if cpu_only else "cuda"
    g = torch.Generator().manual_seed(11)
    for tokens, hidden in ((1, 2560), (64, 2560), (17, 4096), (1, 8192)):
        x = torch.randn(tokens, hidden, generator=g).to(torch.bfloat16).to(device)
        w = (
            (torch.randn(1, hidden, generator=g) / hidden**0.5)
            .to(torch.bfloat16)
            .to(device)
        )
        out = torch.randn(tokens, hidden, generator=g).to(torch.bfloat16).to(device)
        expected = torch_path(x, w, out)
        actual = sg.gated_output(x, w, out)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        # The gate itself must agree to bf16 resolution: compare the ratio on
        # the largest outputs (where a wrong gate would show first).
        big = out.abs().to(torch.float32) > 0.5
        ratio_a = (actual.to(torch.float32) / out.to(torch.float32))[big]
        ratio_e = (expected.to(torch.float32) / out.to(torch.float32))[big]
        torch.testing.assert_close(ratio_a, ratio_e, rtol=1e-2, atol=1e-2)
        print(f"PASS eager tokens={tokens} hidden={hidden}")
    if cpu_only:
        return
    x = torch.randn(1, 2560, generator=g).to(torch.bfloat16).cuda()
    w = (torch.randn(1, 2560, generator=g) / 2560**0.5).to(torch.bfloat16).cuda()
    out = torch.randn(1, 2560, generator=g).to(torch.bfloat16).cuda()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            result = sg.gated_output(x, w, out)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = sg.gated_output(x, w, out)
    for replay in range(3):
        x.copy_(torch.randn(1, 2560, generator=g).to(torch.bfloat16))
        out.copy_(torch.randn(1, 2560, generator=g).to(torch.bfloat16))
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(result, torch_path(x, w, out), rtol=2e-2, atol=2e-2)
        print(f"PASS graph replay {replay}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    run(parser.parse_args().cpu_only)
