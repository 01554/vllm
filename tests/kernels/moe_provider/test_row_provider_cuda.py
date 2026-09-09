# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA checks of RowCacheWeightProvider staged copies (skipped without CUDA).

Two forwards with an eviction between them; the reused slot must hold all
six tensors of the new expert byte-exactly (different global scales), the
previous forward's output (read from the old slot contents before the
eviction) must match its reference so a premature overwrite is caught,
every valid route is planned exactly once, and prepare() from another
stream is rejected.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_row_provider import (
    RowCacheWeightProvider,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

BANK = (
    ("w13", (8, 4), torch.uint8),
    ("w2", (4, 4), torch.uint8),
    ("w13_scale", (8, 1), torch.float8_e4m3fn),
    ("w2_scale", (4, 1), torch.float8_e4m3fn),
    ("w13_scale_2", (2,), torch.float32),
    ("w2_scale_2", (1,), torch.float32),
)


def make_source(experts=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = {}
    for name, shape, dtype in BANK:
        if dtype == torch.uint8:
            t = torch.randint(0, 256, (experts, *shape), generator=g, dtype=torch.uint8)
        elif dtype == torch.float8_e4m3fn:
            t = (torch.rand((experts, *shape), generator=g) * 2 + 0.5).to(dtype)
        else:
            t = torch.rand((experts, *shape), generator=g) * 3 + 0.1  # distinct globals
        src[name] = t
    return src


def slot_bytes(p, slot):
    out = {}
    for name in ("w13", "w2", "w13_scale", "w2_scale", "w13_scale_2", "w2_scale_2"):
        buf = getattr(p, f"buf_{name}")
        out[name] = (
            buf[slot].detach().cpu().contiguous().reshape(-1).view(torch.uint8).clone()
        )
    return out


def src_bytes(src, expert):
    return {
        name: src[name][expert].contiguous().reshape(-1).view(torch.uint8).clone()
        for name in src
    }


def test_eviction_reuse_bytes_and_previous_reader_output():
    src = make_source()
    p = RowCacheWeightProvider(
        2,
        src["w13"],
        src["w2"],
        src["w13_scale"],
        src["w2_scale"],
        w13_scale_2=src["w13_scale_2"],
        w2_scale_2=src["w2_scale_2"],
        device="cuda",
    )
    r1 = p.prepare(torch.tensor([[0, 1]], dtype=torch.int32))
    slot0 = int(r1.expert_map[0])
    owner = torch.cuda.current_stream(p.device)
    # A slow "previous reader" of expert 0's slot on the owner stream, and no
    # host synchronization before the next prepare: the next copies must be
    # ordered behind it by the release event alone.
    big = p.buf_w13[slot0].float().repeat(4096, 1)
    read0 = big
    for _ in range(64):
        read0 = read0 * 1.0 + 0.0
    read0 = read0.sum() / 4096 + p.buf_w13_scale_2[slot0].sum()
    ref0 = src["w13"][0].float().sum() + src["w13_scale_2"][0].sum()
    r2 = p.prepare(
        torch.tensor([[1, 2]], dtype=torch.int32)
    )  # evicts 0 -> expert 2 in slot0
    assert p._copy_stream is not None
    assert p._copy_stream.cuda_stream != owner.cuda_stream  # a separate copy stream
    torch.accelerator.synchronize(p.device)
    assert int(r2.expert_map[0]) == -1 and int(r2.expert_map[2]) == slot0
    got = slot_bytes(p, slot0)
    want = src_bytes(src, 2)
    for name in want:
        assert torch.equal(got[name], want[name]), name
    # the read issued before the eviction saw expert 0, not the overwrite
    assert torch.allclose(read0.cpu(), ref0, rtol=1e-3)
    assert p.stats()["evictions"] == 1 and p.stats()["last_copies"] == 1


def test_all_routes_planned_exactly_once_and_other_stream_rejected():
    src = make_source()
    p = RowCacheWeightProvider(2, src["w13"], src["w2"], device="cuda")
    ids = torch.tensor([[0, 1], [1, 2], [3, -1], [4, 5]], dtype=torch.int32)
    plan = p.plan_chunks(ids)
    covered = []
    for rows, unique in plan:
        r = p.prepare(ids[rows], unique)
        for t in range(*rows.indices(ids.shape[0])):
            for e in ids[t].tolist():
                if e >= 0:
                    assert int(r.expert_map[e]) >= 0
                    covered.append((t, e))
    assert sorted(covered) == sorted(
        (t, int(e)) for t in range(ids.shape[0]) for e in ids[t].tolist() if e >= 0
    )
    other = torch.cuda.Stream(p.device)
    with torch.cuda.stream(other), pytest.raises(RuntimeError):
        p.prepare(torch.tensor([[0, 1]], dtype=torch.int32))
