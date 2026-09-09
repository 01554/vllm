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
    names = ("w13", "w2", "w13_scale", "w2_scale", "w13_scale_2", "w2_scale_2")
    bufs = [getattr(p, f"buf_{n}") for n in names]
    # Pre-allocate outputs and warm the reader so the timed pass allocates
    # nothing (allocator events would otherwise serialize the streams).
    acc = torch.zeros((), dtype=torch.float32, device=p.device)

    def slow_read(acc):
        for _ in range(256):
            for b in bufs:
                acc = acc + b[slot0].reshape(-1).view(torch.uint8).float().sum()
        return acc

    slow_read(acc)
    torch.accelerator.synchronize(p.device)
    reader_done = torch.cuda.Event()
    read0 = slow_read(acc)  # reads all six slot tensors of expert 0
    reader_done.record(owner)
    # The next prepare is enqueued while the reader is still running (no
    # host synchronization in between); the release event must order the
    # copies behind it.
    r2 = p.prepare(torch.tensor([[1, 2]], dtype=torch.int32))  # evicts 0 -> expert 2
    overlapped = not reader_done.query()
    assert p._copy_stream is not None
    assert p._copy_stream.cuda_stream != owner.cuda_stream  # a separate copy stream
    torch.accelerator.synchronize(p.device)
    assert overlapped, (
        "reader finished before the copies were enqueued; test is inconclusive"
    )
    assert int(r2.expert_map[0]) == -1 and int(r2.expert_map[2]) == slot0
    got = slot_bytes(p, slot0)
    want = src_bytes(src, 2)
    for name in want:
        assert torch.equal(got[name], want[name]), name
    ref = torch.zeros((), dtype=torch.float32)
    for _ in range(256):
        for n in names:
            ref = (
                ref + src[n][0].contiguous().reshape(-1).view(torch.uint8).float().sum()
            )
    assert torch.equal(read0.cpu(), ref)  # the reader saw expert 0, never the overwrite
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


def test_invalidate_then_prepare_orders_reuse_behind_the_previous_reader():
    src = make_source()
    p = RowCacheWeightProvider(2, src["w13"], src["w2"], device="cuda")
    r1 = p.prepare(torch.tensor([[0, 1]], dtype=torch.int32))
    slot1 = int(r1.expert_map[1])
    acc = torch.zeros((), dtype=torch.float32, device=p.device)
    for _ in range(256):
        acc = acc + p.buf_w13[slot1].reshape(-1).float().sum()
    p.invalidate(1)  # frees the slot while the reader may still be running
    r2 = p.prepare(torch.tensor([[3, 0]], dtype=torch.int32))  # 3 reuses slot1
    torch.accelerator.synchronize(p.device)
    assert int(r2.expert_map[3]) == slot1
    assert torch.equal(acc.cpu(), (src["w13"][1].reshape(-1).float().sum() * 256))
    assert torch.equal(p.buf_w13[slot1].cpu(), src["w13"][3])
