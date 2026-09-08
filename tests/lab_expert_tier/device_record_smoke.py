# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU smoke for the fused routing record: the Triton program must leave a
CUDA observer in the same state as the torch reference leaves a CPU one,
layer by layer, through finish_step, in eager mode and under graph replay.

    python tests/lab_expert_tier/device_record_smoke.py [--cpu-only]
"""

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_device_record as t  # noqa: E402
from lab_expert_tier import device_record as dr  # noqa: E402
from lab_expert_tier.heat_device import DeviceObserver  # noqa: E402

L, E, K, CAP = 48, 512, 10, 4


def make_observer(device):
    observer = DeviceObserver(num_layers=L, num_experts=E, decay=0.5, sync_period=1)
    observer.allocate(torch.device(device), L, K, CAP, num_experts=E)
    return observer


def forward_inputs(rng, rows, padding, invalid_layer=None, weight_case=None):
    """weight_case: (layer, value) puts `value` into a valid row's weight."""
    layers = []
    for layer in range(L):
        ids = torch.tensor(
            [[rng.randrange(E) for _ in range(K)] for _ in range(rows)],
            dtype=torch.int32,
        )
        weights = torch.tensor(
            [[rng.choice([0.0, 0.25, 1.0]) for _ in range(K)] for _ in range(rows)]
        )
        for r in range(rows):
            if padding[r]:
                ids[r] = -1
        if invalid_layer == layer:
            ids[0, 0] = E + 7
        if weight_case is not None and weight_case[0] == layer:
            weights[0, 0] = weight_case[1]
        if padding.any():
            # Non-finite weights on padding rows must be tolerated.
            weights[padding.nonzero()[0, 0], 0] = float("nan")
        hot_map = torch.tensor(
            [rng.choice([-1, rng.randrange(12000)]) for _ in range(E)],
            dtype=torch.int32,
        )
        layers.append((ids, weights, hot_map))
    return layers


def compare(cpu, gpu, label):
    a, b = t.state(cpu), {k: v.cpu() for k, v in t.state(gpu).items()}
    for name in a:
        assert torch.equal(a[name], b[name]), (label, name)


def run(cpu_only):
    device = "cpu" if cpu_only else "cuda"
    rng = random.Random(7)
    cases = [
        ("batch1", 1, torch.tensor([False]), None, None),
        ("padded", 4, torch.tensor([False, True, False, True]), None, None),
        ("invalid", 2, torch.tensor([False, False]), 5, None),
        ("pos_inf", 2, torch.tensor([False, False]), None, (3, float("inf"))),
        ("neg_inf", 2, torch.tensor([False, False]), None, (9, -float("inf"))),
        ("nan", 2, torch.tensor([False, False]), None, (20, float("nan"))),
        ("negative", 2, torch.tensor([False, False]), None, (47, -0.5)),
    ]
    for label, rows, padding, invalid_layer, weight_case in cases:
        cpu, gpu = make_observer("cpu"), make_observer(device)
        layers = forward_inputs(rng, rows, padding, invalid_layer, weight_case)
        expect_error = invalid_layer is not None or weight_case is not None
        for layer, (ids, weights, hot_map) in enumerate(layers):
            dr.record(cpu.kernel_targets(), layer, rows, ids, weights, padding, hot_map)
            dr.record(
                gpu.kernel_targets(),
                layer,
                rows,
                ids.to(device),
                weights.to(device),
                padding.to(device),
                hot_map.to(device),
            )
            compare(cpu, gpu, (label, "layer", layer))
        cpu.accumulator.finish_step(rows)
        gpu.accumulator.finish_step(rows)
        compare(cpu, gpu, (label, "finish"))
        assert bool(gpu.accumulator._error_flag.cpu()) == expect_error, label
        print(f"PASS eager {label}: rows={rows} layers={L}")
    if cpu_only:
        return
    # Graph replay: capture one full forward on static buffers, replay with
    # new routing three times, and match the CPU reference each time.
    rows = 1
    padding = torch.tensor([False])
    cpu, gpu = make_observer("cpu"), make_observer(device)
    ids_buf = [torch.zeros(rows, K, dtype=torch.int32, device=device) for _ in range(L)]
    w_buf = [torch.zeros(rows, K, device=device) for _ in range(L)]
    map_buf = [torch.zeros(E, dtype=torch.int32, device=device) for _ in range(L)]
    pad_buf = padding.to(device)
    targets = gpu.kernel_targets()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            for layer in range(L):
                dr.record(
                    targets,
                    layer,
                    rows,
                    ids_buf[layer],
                    w_buf[layer],
                    pad_buf,
                    map_buf[layer],
                )
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for layer in range(L):
            dr.record(
                targets,
                layer,
                rows,
                ids_buf[layer],
                w_buf[layer],
                pad_buf,
                map_buf[layer],
            )
    for replay in range(3):
        layers = forward_inputs(rng, rows, padding)
        for layer, (ids, weights, hot_map) in enumerate(layers):
            ids_buf[layer].copy_(ids)
            w_buf[layer].copy_(weights)
            map_buf[layer].copy_(hot_map)
            dr.record(cpu.kernel_targets(), layer, rows, ids, weights, padding, hot_map)
        graph.replay()
        torch.accelerator.synchronize()
        compare(cpu, gpu, ("graph", replay))
        cpu.accumulator.finish_step(rows)
        gpu.accumulator.finish_step(rows)
        compare(cpu, gpu, ("graph-finish", replay))
        print(f"PASS graph replay {replay}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    run(parser.parse_args().cpu_only)
