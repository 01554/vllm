# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-only logical align/remap correctness smoke; run in the candidate image.

Exercises the actual CUDA align operation and runtime remap helpers, not GEMM.
Run after the runtime dispatch fix; this cannot prove every caller dispatches
through the safe path. Full TierLayer verification remains a model-run check.
"""

import torch

from vllm._lab_expert_tier.runtime import mask_routes, physical_block_experts
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)


def run_case(rows, graph_mode):
    block, experts = 8, 512
    ids = torch.full((rows, 10), -1, dtype=torch.int32, device="cuda")
    mapping = torch.full((experts,), -1, dtype=torch.int32, device="cuda")
    mapping[0:10] = torch.arange(550, 560, dtype=torch.int32, device="cuda")
    mapping[20:30] = torch.arange(12000, 12010, dtype=torch.int32, device="cuda")
    result = {}

    def forward():
        routed = mask_routes(ids, mapping)
        sorted_ids, logical, padded = moe_align_block_size(
            routed, block, experts, None, ignore_invalid_experts=True
        )
        result["value"] = (
            sorted_ids,
            physical_block_experts(logical, padded, block, mapping, experts),
            padded,
        )

    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        forward()
    torch.cuda.current_stream().wait_stream(warm)
    torch.accelerator.synchronize()
    graph = None
    if graph_mode:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
    routes = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [20, 21, 20, 29, 0, 1, -1, 100, 20, -1],
        [-1] * 10,
    ]
    for iteration in range(2):
        for route in routes:
            host_ids = torch.tensor([route] * rows, dtype=torch.int32)
            ids.copy_(host_ids)
            # The captured graph must read updated map contents too.
            mapping[0:10].add_(1)
            if graph is None:
                forward()
            else:
                graph.replay()
            torch.accelerator.synchronize()
            sorted_ids, physical, padded = result["value"]
            sorted_cpu = sorted_ids.cpu()
            physical_cpu = physical.cpu()
            map_cpu = mapping.cpu()
            n = int(padded.item())
            assert n % block == 0
            expected = {
                i: int(map_cpu[e])
                for i, e in enumerate(host_ids.flatten().tolist())
                if e >= 0 and map_cpu[e] >= 0
            }
            seen = {}
            for position in range(n):
                token = int(sorted_cpu[position])
                if token >= host_ids.numel():
                    continue
                assert token not in seen, (rows, iteration, token)
                seen[token] = int(physical_cpu[position // block])
            assert seen == expected, (rows, graph_mode, seen, expected)
            assert bool((physical_cpu[n // block :] == -1).all())
    print(
        f"PASS align rows={rows} graph={graph_mode}: "
        "physical550+, staging12000+, duplicate/padding/absent/map-update"
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script must be run by the GPU owner")
    for rows in (1, 4):
        for graph_mode in (False, True):
            run_case(rows, graph_mode)
