# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alternate captured reference alignment+remap and small alignment.

Run only after the expert_pool correctness suite passes. This measures alignment
alone, excluding planner/copy/consumer/PLE, and is not a serving speed claim.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.expert_pool.layer import (
    physical_block_experts_device,
)
from vllm.model_executor.layers.fused_moe.expert_pool.small_align import (
    _small_align,
    small_align,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import triton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for lanes in (10, 64):
        for pattern in ("distinct", "duplicate", "padding"):
            ids = torch.arange(lanes, device="cuda", dtype=torch.int64)
            if pattern == "duplicate":
                ids.fill_(1)
            elif pattern == "padding":
                ids.fill_(-1)
            mapping = torch.arange(512, device="cuda", dtype=torch.int32) + 512
            block = 16

            def reference(ids=ids, mapping=mapping, block=block):
                sorted_ids, logical, count = moe_align_block_size(
                    ids, block, 512, None, ignore_invalid_experts=True
                )
                physical = physical_block_experts_device(
                    logical, count, block, mapping, 512
                )
                return sorted_ids, physical, count

            def candidate(ids=ids, mapping=mapping, block=block):
                return small_align(ids, mapping, block, 1024)

            graphs, outputs = [], []
            for fn in (reference, candidate):
                for _ in range(3):
                    fn()
                torch.accelerator.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs.append(fn())
                graphs.append(graph)
            compiled = _small_align[(1,)](
                ids,
                mapping,
                *outputs[1],
                lanes,
                512,
                1024,
                block,
                triton.next_power_of_2(lanes),
                triton.next_power_of_2(lanes * block),
                num_warps=4,
            )
            for extension in ("ptx", "cubin"):
                artifact = compiled.asm[extension]
                path = args.output / f"lanes{lanes}-{pattern}.{extension}"
                path.write_bytes(
                    artifact.encode() if isinstance(artifact, str) else artifact
                )
            (args.output / f"lanes{lanes}-{pattern}-resources.json").write_text(
                json.dumps(dict(n_regs=compiled.n_regs, n_spills=compiled.n_spills))
            )
            samples = [[], []]
            for batch in range(30):
                for index in (0, 1) if batch % 2 == 0 else (1, 0):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100):
                        graphs[index].replay()
                    end.record()
                    end.synchronize()
                    samples[index].append(start.elapsed_time(end) * 1000 / 100)
            records.append(
                dict(
                    lanes=lanes,
                    pattern=pattern,
                    block=block,
                    samples_us=samples,
                    medians_us=[statistics.median(s) for s in samples],
                )
            )
            # Retain inputs, outputs and both graphs throughout every batch.
            assert len(outputs) == len(graphs) == 2
    source = Path(__file__).resolve().parents[2] / (
        "vllm/model_executor/layers/fused_moe/expert_pool/small_align.py"
    )
    (args.output / "result.json").write_text(
        json.dumps(
            dict(
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
