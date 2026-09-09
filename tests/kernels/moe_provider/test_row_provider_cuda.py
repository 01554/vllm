# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA checks of RowCacheWeightProvider staged copies (skipped without CUDA).

Two forwards with an eviction between them; the reused slot must hold all
six tensors of the new expert byte-exactly (different global scales), the
previous forward's output (read from the old slot contents before the
eviction) must match its reference so a premature overwrite is caught,
every valid route is planned exactly once, and prepare() from another
stream takes ownership while the previous stream's reader is still ordered
ahead of the copies.

The reader is a pre-captured CUDA graph (a finite GPU delay followed by a
read of the real slot tensors) replayed on the owner stream, so it is still
running when the next prepare()/invalidate() is enqueued from the host. The
graph is only a test fixture; it is not evidence of graph support in the
provider.
"""

import time

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


DELAY_N = 4096
DELAY_MATMULS = 8
REPLAYS = 100


class GraphReader:
    """Captured graph: GPU delay, then sum the bytes of the given slot tensors.

    All buffers and the graph are prepared before the measured region; replay()
    enqueues on the current (owner) stream and the accumulator holds the sum
    seen by the last replay.
    """

    def __init__(self, tensors, device):
        self.acc = torch.zeros((), dtype=torch.float32, device=device)
        self.a = torch.full((DELAY_N, DELAY_N), 1e-3, device=device)
        self.b = torch.empty_like(self.a)
        self.tensors = tensors

        def body():
            for _ in range(DELAY_MATMULS):  # constant input, no feedback
                torch.matmul(self.a, self.a, out=self.b)
            self.acc.zero_()
            for t in self.tensors:
                self.acc.add_(t.reshape(-1).view(torch.uint8).float().sum())

        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(3):
                body()
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=side):
            body()
        torch.accelerator.synchronize(device)

    def replay(self, n=REPLAYS):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        for _ in range(n):
            self.graph.replay()
        self.end.record()
        return self.acc

    def gpu_ms(self):
        return self.start.elapsed_time(self.end)


def inconclusive(host_ms, reader):
    return (
        "reader finished before the copies were enqueued; test is inconclusive "
        f"(prepare host {host_ms:.1f} ms, reader GPU {reader.gpu_ms():.1f} ms: "
        "host >= reader means prepare blocked on the GPU)"
    )


def ref_sum(tensors):
    ref = torch.zeros((), dtype=torch.float32)
    for t in tensors:
        ref = ref + t.contiguous().reshape(-1).view(torch.uint8).float().sum()
    return ref


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
    ref = ref_sum([src[n][0] for n in names])
    assert not torch.equal(ref, ref_sum([src[n][2] for n in names]))
    reader = GraphReader([getattr(p, f"buf_{n}")[slot0] for n in names], p.device)
    reader_done = torch.cuda.Event()
    read0 = reader.replay()  # reads all six slot tensors of expert 0
    reader_done.record(owner)
    # The next prepare is enqueued while the reader is still running (no
    # host synchronization in between); the release event must order the
    # copies behind it. Evicts 0 -> expert 2.
    t0 = time.perf_counter()
    r2 = p.prepare(torch.tensor([[1, 2]], dtype=torch.int32))
    host_ms = (time.perf_counter() - t0) * 1e3
    overlapped = not reader_done.query()
    assert p._copy_stream is not None
    assert p._copy_stream.cuda_stream != owner.cuda_stream  # a separate copy stream
    torch.accelerator.synchronize(p.device)
    assert overlapped, inconclusive(host_ms, reader)
    assert int(r2.expert_map[0]) == -1 and int(r2.expert_map[2]) == slot0
    got = slot_bytes(p, slot0)
    want = src_bytes(src, 2)
    for name in want:
        assert torch.equal(got[name], want[name]), name
    assert torch.equal(read0.cpu(), ref)  # the reader saw expert 0, never the overwrite
    assert p.stats()["evictions"] == 1 and p.stats()["last_copies"] == 1


def test_all_routes_planned_exactly_once_and_other_stream_takes_ownership():
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
    with torch.cuda.stream(other):
        r = p.prepare(torch.tensor([[0, 1]], dtype=torch.int32))
    torch.accelerator.synchronize(p.device)
    assert int(r.expert_map[0]) >= 0 and int(r.expert_map[1]) >= 0
    assert p.stats()["owner_changes"] == 1
    assert p._owner_stream is not None
    assert p._owner_stream.cuda_stream == other.cuda_stream


def test_owner_change_orders_copies_behind_the_previous_stream_reader():
    """A reader still running on the previous owner stream (profile run,
    capture stream, replay stream) must finish before a prepare() issued
    from another stream rewrites its slot."""
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
    first = torch.cuda.Stream(p.device)
    with torch.cuda.stream(first):
        r1 = p.prepare(torch.tensor([[0, 1]], dtype=torch.int32))
    torch.accelerator.synchronize(p.device)
    slot0 = int(r1.expert_map[0])
    names = ("w13", "w2", "w13_scale", "w2_scale", "w13_scale_2", "w2_scale_2")
    ref = ref_sum([src[n][0] for n in names])
    assert not torch.equal(ref, ref_sum([src[n][2] for n in names]))
    reader = GraphReader([getattr(p, f"buf_{n}")[slot0] for n in names], p.device)
    reader_done = torch.cuda.Event()
    with torch.cuda.stream(first):  # the reader runs on the previous owner
        read0 = reader.replay()
        reader_done.record(first)
    second = torch.cuda.Stream(p.device)
    with torch.cuda.stream(second):  # evicts 0 -> expert 2 from another stream
        t0 = time.perf_counter()
        r2 = p.prepare(torch.tensor([[1, 2]], dtype=torch.int32))
        host_ms = (time.perf_counter() - t0) * 1e3
    overlapped = not reader_done.query()
    torch.accelerator.synchronize(p.device)
    assert overlapped, inconclusive(host_ms, reader)
    assert p.stats()["owner_changes"] == 1
    assert int(r2.expert_map[2]) == slot0
    got = slot_bytes(p, slot0)
    want = src_bytes(src, 2)
    for name in want:
        assert torch.equal(got[name], want[name]), name
    assert torch.equal(read0.cpu(), ref)  # the reader saw expert 0, never the overwrite
    # The shared resident map holds the final generation: the previous
    # owner's upload cannot land after the new stream's.
    assert p.expert_map.cpu().tolist() == p._map_host


def test_invalidate_from_another_stream_keeps_the_resident_map_ordered():
    """invalidate() on stream B while A's reader (and A's map upload) are
    queued, then prepare() on B: the resident map must end in the final
    generation and the reuse must wait for A's reader."""
    src = make_source()
    p = RowCacheWeightProvider(2, src["w13"], src["w2"], device="cuda")
    first = torch.cuda.Stream(p.device)
    with torch.cuda.stream(first):
        r1 = p.prepare(torch.tensor([[0, 1]], dtype=torch.int32))
    torch.accelerator.synchronize(p.device)
    slot1 = int(r1.expert_map[1])
    assert not torch.equal(ref_sum([src["w13"][1]]), ref_sum([src["w13"][3]]))
    reader = GraphReader([p.buf_w13[slot1]], p.device)
    reader_done = torch.cuda.Event()
    with torch.cuda.stream(first):
        read1 = reader.replay()
        reader_done.record(first)
    second = torch.cuda.Stream(p.device)
    t0 = time.perf_counter()
    with torch.cuda.stream(second):
        p.invalidate(1)  # map write goes to the owner stream, behind the reader
        r2 = p.prepare(torch.tensor([[3, 0]], dtype=torch.int32))  # 3 reuses slot1
    host_ms = (time.perf_counter() - t0) * 1e3
    overlapped = not reader_done.query()
    torch.accelerator.synchronize(p.device)
    assert overlapped, inconclusive(host_ms, reader)
    assert int(r2.expert_map[3]) == slot1
    assert torch.equal(read1.cpu(), ref_sum([src["w13"][1]]))
    assert torch.equal(p.buf_w13[slot1].cpu(), src["w13"][3])
    assert p.expert_map.cpu().tolist() == p._map_host


def test_invalidate_then_prepare_orders_reuse_behind_the_previous_reader():
    src = make_source()
    p = RowCacheWeightProvider(2, src["w13"], src["w2"], device="cuda")
    r1 = p.prepare(torch.tensor([[0, 1]], dtype=torch.int32))
    slot1 = int(r1.expert_map[1])
    owner = torch.cuda.current_stream(p.device)
    assert not torch.equal(ref_sum([src["w13"][1]]), ref_sum([src["w13"][3]]))
    reader = GraphReader([p.buf_w13[slot1]], p.device)
    reader_done = torch.cuda.Event()
    read1 = reader.replay()
    reader_done.record(owner)
    t0 = time.perf_counter()
    p.invalidate(1)  # frees the slot; may not wait for the reader
    r2 = p.prepare(torch.tensor([[3, 0]], dtype=torch.int32))  # 3 reuses slot1
    host_ms = (time.perf_counter() - t0) * 1e3
    overlapped = not reader_done.query()
    torch.accelerator.synchronize(p.device)
    assert overlapped, inconclusive(host_ms, reader)
    assert int(r2.expert_map[3]) == slot1
    assert torch.equal(read1.cpu(), ref_sum([src["w13"][1]]))
    assert torch.equal(p.buf_w13[slot1].cpu(), src["w13"][3])
