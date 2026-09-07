# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Global pool six-bank UVA/reference smoke. GPU execution: integration owner.

Run in the candidate image, or --cpu-only for reference scenario validation.
This checks copies/planner/maps; global_pool_align_smoke.py checks CUDA align.
"""

import argparse
import dataclasses
import importlib.util
import sys
from pathlib import Path

import torch


def load_pool(repo):
    # Isolated package permits CPU reference validation without a built vLLM.
    path = repo / "vllm" / "_lab_expert_tier"
    spec = importlib.util.spec_from_file_location(
        "pool_smoke_lab", path / "__init__.py", submodule_search_locations=[str(path)]
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from pool_smoke_lab import global_pool

    return global_pool


def tensor_fields(obj):
    return {
        f.name: getattr(obj, f.name)
        for f in dataclasses.fields(obj)
        if isinstance(getattr(obj, f.name), torch.Tensor)
    }


def run_case(gp, graph_mode, cpu_only):
    sources = []
    for layer in range(3):
        sources.append(
            {
                name: (
                    torch.arange(512, dtype=torch.int32)[:, None] * 1000
                    + torch.arange(width, dtype=torch.int32)[None, :]
                    + layer * 1000000
                    + index * 10000000
                ).contiguous()
                for index, (name, width) in enumerate(
                    zip(gp.TENSORS, (33, 65, 3, 17, 1, 7))
                )
            }
        )
    cpu = gp.GlobalPool(torch.device("cpu"), sources[0], [250] * 3, 10)
    for layer in range(3):
        for name in gp.TENSORS:
            cpu.bank[name][layer * 250 : (layer + 1) * 250].copy_(
                sources[layer][name][:250]
            )
    cpu.tables.last_use[:1074] = 1
    cpu.tables.clock.fill_(1)
    cb = [gp.allocate_step_buffers(torch.device("cpu"), 512, 16) for _ in range(3)]
    original = [{n: t.clone() for n, t in s.items()} for s in sources]
    gpu = None
    graphs = {}
    if not cpu_only:
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        pinned = [{n: t.pin_memory() for n, t in s.items()} for s in sources]
        uva = [
            {n: get_accelerator_view_from_cpu_tensor(t) for n, t in s.items()}
            for s in pinned
        ]
        gpu = gp.GlobalPool(torch.device("cuda"), sources[0], [250] * 3, 10)
        gb = [gp.allocate_step_buffers(torch.device("cuda"), 512, 16) for _ in range(3)]
        ids_device = [
            torch.full((1, 10), -1, dtype=torch.int32, device="cuda") for _ in range(3)
        ]

        def forward(layer):
            gp.step(gpu.tables, layer, ids_device[layer], gb[layer])
            gp.copy_in(uva[layer], gpu.bank, gb[layer])

        # Compile/capture with a closed gate, then restore all reference state.
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for layer in range(3):
                forward(layer)
        torch.cuda.current_stream().wait_stream(warm)
        torch.accelerator.synchronize()
        if graph_mode:
            for layer in range(3):
                graphs[layer] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[layer]):
                    forward(layer)
        torch.accelerator.synchronize()
        for name, value in tensor_fields(cpu.tables).items():
            getattr(gpu.tables, name).copy_(value)
        for name in gp.TENSORS:
            gpu.bank[name].copy_(cpu.bank[name])

    sequence = [
        (False, 0, list(range(500, 510)), "staging750"),
        (True, 0, list(range(500, 510)), "physical550"),
        (True, 2, list(range(50, 60)), "cross-layer-return"),
        (True, 0, [500, 500, 501, -1, 502, 509, -1, 500, 508, 508], "duplicates"),
        (True, 1, [-1] * 10, "padding"),
    ]
    for repeat in range(3):
        for enabled, layer, ids, label in sequence:
            host_ids = torch.tensor([ids], dtype=torch.int32)
            gp.set_gate(cpu.tables, enabled)
            gp.step(cpu.tables, layer, host_ids, cb[layer])
            gp.copy_in(sources[layer], cpu.bank, cb[layer])
            cpu.snapshot()
            if repeat == 0 and label == "physical550":
                assert cpu.tables.hot_phys[500:510].tolist() == list(range(550, 560))
            if label == "staging750" and repeat == 0:
                assert cb[layer].gather_dst[:10].tolist() == list(range(750, 760))
            if gpu is not None:
                gp.set_gate(gpu.tables, enabled)
                ids_device[layer].copy_(host_ids)
                if graph_mode:
                    graphs[layer].replay()
                else:
                    forward(layer)
                torch.accelerator.synchronize()
                gpu.snapshot()
                for name, value in tensor_fields(cpu.tables).items():
                    assert torch.equal(getattr(gpu.tables, name).cpu(), value), (
                        label,
                        name,
                    )
                for name in gp.TENSORS:
                    assert torch.equal(gpu.bank[name].cpu(), cpu.bank[name]), (
                        label,
                        name,
                    )
                for name in (
                    "step_map",
                    "gather_count",
                    "promoted_count",
                    "staged_count",
                ):
                    assert torch.equal(
                        getattr(gb[layer], name).cpu(), getattr(cb[layer], name)
                    ), (label, name)
                for count_name, names in (
                    ("gather_count", ("gather_src", "gather_dst")),
                    ("staged_count", ("staged_expert", "staged_row")),
                ):
                    count = int(getattr(cb[layer], count_name)[0])
                    for name in names:
                        assert torch.equal(
                            getattr(gb[layer], name)[:count].cpu(),
                            getattr(cb[layer], name)[:count],
                        ), (label, name)
    for layer in range(3):
        for name in gp.TENSORS:
            assert torch.equal(sources[layer][name], original[layer][name])
            if gpu is not None:
                assert torch.equal(pinned[layer][name], original[layer][name])
    print(
        f"PASS pool graph={graph_mode} cpu_only={cpu_only}: "
        "15 steps, six banks, immutable RAM, "
        "physical550/staging750/cross-layer/padding"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    module = load_pool(args.repo)
    if args.cpu_only:
        run_case(module, False, True)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; run only as the GPU integration owner")
        for mode in (False, True):
            run_case(module, mode, False)
