# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU check of the staged-copy event order with the CUDA stream API stubbed.

Contract (per prepare on CUDA): release event recorded on the owner
compute stream -> copy stream waits on it -> slot fills on the copy stream
-> ready event recorded on the copy stream -> owner stream waits on it.
"""

from contextlib import contextmanager
from unittest import mock

import torch

from vllm.model_executor.layers.fused_moe.expert_row_provider import (
    RowCacheWeightProvider,
)

LOG: list[str] = []


class _Stream:
    def __init__(self, device=None, name="copy"):
        self.name = name
        self.cuda_stream = 0x100 if name == "owner" else 0x200

    def wait_event(self, event):
        LOG.append(f"{self.name}.wait({event.name})")

    def synchronize(self):
        LOG.append(f"{self.name}.synchronize")


class _Event:
    count = 0

    def __init__(self):
        _Event.count += 1
        self.name = "release" if _Event.count % 2 == 1 else "ready"

    def record(self, stream):
        LOG.append(f"{self.name}.record({stream.name})")


_OWNER = _Stream(name="owner")
_ACTIVE = {"stream": _OWNER}


@contextmanager
def _use_stream(stream):
    prev = _ACTIVE["stream"]
    _ACTIVE["stream"] = stream
    LOG.append(f"enter({stream.name})")
    try:
        yield
    finally:
        LOG.append(f"exit({stream.name})")
        _ACTIVE["stream"] = prev


def _current_stream(device=None):
    return _ACTIVE["stream"]


def _make():
    experts = 4
    w13 = torch.arange(experts * 8, dtype=torch.float32).reshape(experts, 4, 2)
    w2 = torch.zeros((experts, 2, 4))
    return RowCacheWeightProvider(2, w13, w2, device="cpu")


def test_event_order_on_prepare_and_invalidate():
    LOG.clear()
    _Event.count = 0
    p = _make()
    # Pretend the cache lives on CUDA so the staged path runs; copies stay
    # on CPU tensors (copy_ with non_blocking is fine there).
    p.device = torch.device("cuda")
    with (
        mock.patch.object(
            torch.cuda, "Stream", lambda device=None: _Stream(name="copy")
        ),
        mock.patch.object(torch.cuda, "Event", _Event),
        mock.patch.object(torch.cuda, "current_stream", _current_stream),
        mock.patch.object(torch.cuda, "stream", _use_stream),
    ):
        p.prepare(torch.tensor([[0, 1]], dtype=torch.int32))
        first = list(LOG)
        LOG.clear()
        p.prepare(torch.tensor([[1, 2]], dtype=torch.int32))  # evicts 0
        second = list(LOG)
        LOG.clear()
        p.invalidate(2)
        third = list(LOG)
    for seq in (first, second):
        assert seq[:2] == ["release.record(owner)", "copy.wait(release)"], seq
        assert seq[2] == "enter(copy)" and "exit(copy)" in seq, seq
        assert seq[-2:] == ["ready.record(copy)", "owner.wait(ready)"], seq
    assert third == ["copy.synchronize"], (
        third
    )  # copies only; reader ordering via the next release event
    assert p.stats()["last_copies"] == 1 and p.stats()["evictions"] == 1
