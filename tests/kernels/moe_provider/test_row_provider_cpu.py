# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the RowCacheWeightProvider surface and placement."""

import torch

from vllm.model_executor.layers.fused_moe.expert_row_provider import (
    RowCacheWeightProvider,
)


def make_provider(capacity=2, experts=4):
    w13 = torch.arange(experts * 8, dtype=torch.float32).reshape(experts, 4, 2)
    w2 = -torch.arange(experts * 8, dtype=torch.float32).reshape(experts, 2, 4)
    s13 = torch.ones((experts, 4, 1))
    s2 = torch.ones((experts, 2, 1))
    g13 = torch.arange(experts, dtype=torch.float32).reshape(experts, 1).repeat(1, 2)
    g2 = torch.arange(experts, dtype=torch.float32)
    return RowCacheWeightProvider(
        capacity, w13, w2, s13, s2, w13_scale_2=g13, w2_scale_2=g2
    ), (w13, w2)


def test_prepare_places_rows_and_hides_the_rest():
    p, (w13, w2) = make_provider()
    ids = torch.tensor([[1, 3]], dtype=torch.int32)
    r = p.prepare(ids)
    resident = {int(e) for e in torch.nonzero(r.expert_map >= 0).flatten()}
    assert resident == {1, 3}
    for e in (1, 3):
        slot = int(r.expert_map[e])
        assert torch.equal(r.w1[slot], w13[e])
        assert torch.equal(r.w2[slot], w2[e])
        assert float(p.buf_w13_scale_2[slot, 0]) == e
        assert float(p.buf_w2_scale_2[slot]) == e
    assert p.stats()["misses"] == 2 and p.stats()["hits"] == 0


def test_eviction_protects_experts_needed_in_this_forward():
    p, (w13, _) = make_provider(capacity=2)
    p.prepare(torch.tensor([[0, 1]], dtype=torch.int32))
    r = p.prepare(torch.tensor([[1, 2]], dtype=torch.int32))
    assert int(r.expert_map[0]) == -1  # evicted
    assert int(r.expert_map[1]) >= 0 and int(r.expert_map[2]) >= 0
    assert torch.equal(r.w1[int(r.expert_map[2])], w13[2])
    s = p.stats()
    assert (s["hits"], s["misses"], s["evictions"]) == (1, 3, 1)


def test_plan_chunks_and_groups_cover_every_route_exactly_once():
    p, _ = make_provider(capacity=2)
    ids = torch.tensor([[0, 1], [1, 2], [3, -1]], dtype=torch.int32)
    plan = p.plan_chunks(ids)
    rows = [list(range(*s.indices(ids.shape[0]))) for s, _ in plan]
    assert sorted(sum(rows, [])) == [0, 1, 2]
    for s, unique in plan:
        chunk = {int(e) for e in ids[s].reshape(-1).tolist() if e >= 0}
        assert chunk == set(unique) and len(unique) <= 2
    groups = p.plan_expert_groups(ids)
    assert sorted(sum(groups, [])) == [0, 1, 2, 3]
    assert all(len(g) <= 2 for g in groups)


def test_overflow_and_invalidate():
    p, _ = make_provider(capacity=2)
    try:
        p.prepare(torch.tensor([[0, 1, 2]], dtype=torch.int32))
        raise AssertionError("expected overflow error")
    except RuntimeError:
        pass
    p.prepare(torch.tensor([[0, 1]], dtype=torch.int32))
    p.invalidate(0)
    assert int(p.expert_map[0]) == -1 and p.stats()["resident"] == 1
