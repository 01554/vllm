# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run actual native NVFP4 arithmetic and replay checks on the GPU owner host.

Run from the repository root: python tests/lab_expert_tier/native_nvfp4_smoke.py.
This checks arithmetic against an independent dense CPU oracle, not performance.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2] / "vllm" / "_lab_expert_tier"
spec = importlib.util.spec_from_file_location(
    "lab_expert_tier",
    ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
assert spec is not None and spec.loader is not None
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from lab_expert_tier import native_nvfp4 as native  # noqa: E402


def dense(bank, prefix):
    """Unpack low-nibble-first FP4 without using the backend's reference."""
    packed = bank[f"{prefix}_weight"].cpu().to(torch.int64)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    scale = bank[f"{prefix}_weight_scale"].cpu().float().repeat_interleave(16, -1)
    return lut[codes] * scale


def oracle(x, weights, ids, bank, mapping):
    gu, down = dense(bank, "w13"), dense(bank, "w2")
    gu_global = bank["w13_weight_scale_2"].cpu().float()
    down_global = bank["w2_weight_scale_2"].cpu().float()
    output = torch.zeros_like(x, device="cpu")
    for token in range(x.shape[0]):
        total = torch.zeros(x.shape[1], dtype=torch.float32)
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            if expert == -1:
                continue
            row = int(mapping[expert])
            hidden = (gu[row] @ x[token].cpu().float() * gu_global[row]).bfloat16()
            gate, up = hidden.float().chunk(2)
            act = (torch.nn.functional.silu(gate) * up).bfloat16()
            value = (
                down[row] @ act.float() * down_global[row] * weights[token, route].cpu()
            ).bfloat16()
            total += value.float()
        output[token] = total.bfloat16()
    return output


def run_case(
    tokens,
    hidden,
    intermediate,
    *,
    grouped=False,
    physical_rows=4,
    uva=False,
    routes_ready=False,
):
    generator = torch.Generator().manual_seed(513)
    bank = {}
    for prefix, n, k in (
        ("w13", 2 * intermediate, hidden),
        ("w2", hidden, intermediate),
    ):
        bank[f"{prefix}_weight"] = torch.randint(
            0,
            256,
            (physical_rows, n, k // 2),
            dtype=torch.uint8,
            generator=generator,
        ).cuda()
        bank[f"{prefix}_weight_scale"] = (
            (
                torch.randint(
                    1, 4, (physical_rows, n, k // 16), generator=generator
                ).float()
                / 8
            )
            .to(torch.float8_e4m3fn)
            .cuda()
        )
        bank[f"{prefix}_weight_scale_2"] = (
            (torch.rand((physical_rows, n), generator=generator) / 8 + 0.03125)
            .half()
            .cuda()
        )
    if uva:
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        # Keep the owning pinned tensors alive through every eager/graph launch.
        backing = {name: tensor.cpu().pin_memory() for name, tensor in bank.items()}
        bank = {
            name: get_accelerator_view_from_cpu_tensor(tensor)
            for name, tensor in backing.items()
        }
    x = (torch.randn(tokens, hidden, generator=generator) / 8).bfloat16().cuda()
    ids = torch.tensor([[0, 2, 0, -1]] * tokens, dtype=torch.int32, device="cuda")
    weights = torch.tensor([[0.125, 0.5, 0.25, 99.0]] * tokens, device="cuda")
    base_row = physical_rows - 4
    mapping = torch.tensor(
        [base_row + 2, base_row, base_row + 3, base_row + 1, -1],
        dtype=torch.int32,
        device="cuda",
    )
    operation, allocator = native.gemv, native.allocate_workspace
    if grouped:
        from lab_expert_tier.native_prefill import allocate_workspace, prefill

        operation, allocator = prefill, allocate_workspace
    workspace = allocator(bank, tokens, 4, num_experts=4)
    if routes_ready:
        if grouped:
            raise ValueError("routes_ready is only supported by the decode smoke")
        ready_routes = torch.tensor(
            [[2, -1, 3, 0]], dtype=torch.int32, device="cuda"
        ).repeat(tokens, 1)
        workspace.routes[:tokens].copy_(ready_routes)
        # Deliberately make the logical inputs unusable.  The ready route
        # buffer must be the only route source in this mode.
        ids.fill_(99)
        mapping.fill_(-1)
        ready_mapping = torch.arange(physical_rows, dtype=torch.int32, device="cuda")
    else:
        ready_mapping = None

    def forward():
        if routes_ready:
            return operation(
                x,
                weights,
                ids,
                bank,
                mapping,
                workspace,
                routes_ready=True,
            )
        return operation(x, weights, ids, bank, mapping, workspace)

    forward()  # Compile and initialize all kernels before capture.
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    for step in range(3):
        if step == 1:
            if routes_ready:
                workspace.routes[:tokens].copy_(
                    torch.tensor(
                        [[1, 0, 2, -1]], dtype=torch.int32, device="cuda"
                    ).repeat(tokens, 1)
                )
                ids.fill_(98)
            else:
                ids[:, 0] = -1  # Formerly valid lanes must overwrite stale scratch.
                mapping[:4].copy_(
                    torch.tensor(
                        [base_row + 1, base_row + 3, base_row, base_row + 2],
                        device="cuda",
                    )
                )
        elif step == 2:
            if routes_ready:
                workspace.routes[:tokens].copy_(
                    torch.tensor(
                        [[3, 2, -1, 0]], dtype=torch.int32, device="cuda"
                    ).repeat(tokens, 1)
                )
                ids.fill_(97)
                bank["w2_weight"][3].zero_()
            else:
                ids[:, 0] = 1
                bank["w2_weight"][
                    base_row + 3
                ].zero_()  # Replay must read live bank bytes.
        graph.replay()
        actual = captured.cpu()
        expected_ids = workspace.routes[:tokens].cpu() if routes_ready else ids.cpu()
        expected_map = ready_mapping.cpu() if routes_ready else mapping.cpu()
        expected = oracle(x, weights, expected_ids, bank, expected_map)
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.0002)
        eager = forward().cpu()
        torch.testing.assert_close(eager, actual, rtol=0, atol=0)
        assert workspace.error.item() == 0
    workspace.error.zero_()
    ids.fill_(-1)
    x.fill_(float("nan"))
    weights.fill_(float("nan"))
    if routes_ready:
        workspace.routes[:tokens].fill_(-1)
    graph.replay()
    assert torch.equal(captured.cpu(), torch.zeros_like(captured.cpu()))
    assert workspace.error.item() == 0
    if routes_ready:
        # Each physical-row failure is checked independently after resetting
        # the sticky flag; padding remains a non-error route.
        workspace.error.zero_()
        workspace.routes[:tokens].fill_(-1)
        workspace.routes[:, 0] = physical_rows
        graph.replay()
        assert torch.equal(captured.cpu(), torch.zeros_like(captured.cpu()))
        assert workspace.error.item() == 1

        workspace.error.zero_()
        workspace.routes[:tokens].fill_(-1)
        workspace.routes[:, 0] = -2
        graph.replay()
        assert torch.equal(captured.cpu(), torch.zeros_like(captured.cpu()))
        assert workspace.error.item() == 1
    else:
        workspace.error.zero_()
        ids[:, 0] = 4  # Out-of-domain ID must never read a physical bank row.
        graph.replay()
        assert torch.equal(captured.cpu(), torch.zeros_like(captured.cpu()))
        assert workspace.error.item() == 1

        workspace.error.zero_()
        # A valid logical expert with no bank row must also zero NaN-valued routes.
        ids.fill_(-1)
        ids[:, 0] = 1
        mapping[1] = -1
        graph.replay()
        assert torch.equal(captured.cpu(), torch.zeros_like(captured.cpu()))
        assert workspace.error.item() == 1

        for invalid_id, invalid_row in ((-2, 0), (1, physical_rows)):
            workspace.error.zero_()
            ids.fill_(-1)
            ids[:, 0] = invalid_id
            mapping[1] = invalid_row
            graph.replay()
            assert torch.equal(captured.cpu(), torch.zeros_like(captured.cpu()))
            assert workspace.error.item() == 1

    # A padding-only replay does not clear a sticky failure by itself.
    if routes_ready:
        workspace.routes[:tokens].fill_(-1)
    else:
        ids.fill_(-1)
    graph.replay()
    assert torch.equal(captured.cpu(), torch.zeros_like(captured.cpu()))
    assert workspace.error.item() == 1
    workspace.error.zero_()
    graph.replay()
    assert workspace.error.item() == 0
    print(
        f"PASS M={tokens}, H={hidden}, I={intermediate}, "
        f"rows={physical_rows}, UVA={uva}: eager/graph/oracle"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped", action="store_true")
    args = parser.parse_args()
    if args.grouped:
        # Both FT prefill tile configurations, partial tiles, and pool row > E.
        run_case(17, 32, 16, grouped=True)
        run_case(65, 128, 32, grouped=True, physical_rows=560)
        run_case(17, 64, 32, grouped=True, uva=True)
    else:
        run_case(1, 32, 16, physical_rows=560)
        run_case(3, 32, 16)
        run_case(1, 2064, 32)  # FreeToken deep-K launch configuration.
        run_case(1, 32, 16, routes_ready=True)
