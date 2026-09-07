# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous expert exchanges that overlap the next forward.

The synchronous path exchanges rows in place at the resync boundary while
the GPU waits. This module moves the copies off the critical path:

- Each layer keeps a ring of spare VRAM rows at the end of its bank and a
  ring of spare pinned RAM rows at the end of its cold bank. A promoted
  cold expert is copied into a spare VRAM row and the evicted hot expert
  into a spare RAM row, on a migration stream, while the next forward
  still runs on the old placement (which only reads the old rows).
- At the next boundary the transfers are complete (or waited for), the
  layer's logical-to-physical row tables flip, the device maps are
  republished, and the policy commits. Retired rows go back to their rings
  tagged with a fence recorded on the compute stream after the last
  forward that used the old map; a later transfer into such a row waits
  for that fence first.

Only plans that fit every layer's free spare rows and reuse no hot or
cold slot within a layer are taken asynchronously; any other plan takes
the synchronous path unchanged, in its original order (see `preflight`).
The logical maps the policy sees never change meaning; the physical row
behind a logical slot does.

The stream and event helpers are the only CUDA touch points; they are
plain functions so CPU tests can substitute incomplete events and fences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Preflight:
    """Whether a whole plan may run asynchronously, and why not."""

    eligible: bool
    reason: str | None
    per_layer: dict[int, tuple[Any, ...]]


def preflight(swaps, budgets):
    """Decide the whole plan at once, before any transfer starts.

    Args:
        swaps: the plan's swaps in order.
        budgets: layer index -> free spare rows (the smaller of the VRAM
            and RAM rings) for that layer.

    Returns:
        Preflight. Ineligible reasons: "empty", "slot_reuse" (a hot or cold
        slot appears twice within one layer, so a later swap would need an
        unpublished result), "over_budget" (a layer has more swaps than free
        spare rows; nothing is truncated).
    """
    per_layer: dict[int, list[Any]] = {}
    for swap in swaps:
        per_layer.setdefault(swap.layer, []).append(swap)
    frozen = {layer: tuple(items) for layer, items in per_layer.items()}
    if not frozen:
        return Preflight(False, "empty", frozen)
    for layer, items in frozen.items():
        hot = [swap.hot_slot for swap in items]
        cold = [swap.cold_slot for swap in items]
        if len(set(hot)) != len(hot) or len(set(cold)) != len(cold):
            return Preflight(False, "slot_reuse", frozen)
        if len(items) > budgets.get(layer, 0):
            return Preflight(False, "over_budget", frozen)
    return Preflight(True, None, frozen)


@dataclass(frozen=True)
class SpareRow:
    """A free physical row; `fence` guards its last reader, if any."""

    row: int
    fence: Any = None


class SpareRing:
    """FIFO of spare rows. Rows come back only after their reader's fence."""

    def __init__(self, rows):
        self._rows: list[SpareRow] = [SpareRow(row) for row in rows]

    @property
    def free(self):
        return len(self._rows)

    def pop(self):
        if not self._rows:
            raise RuntimeError("Spare ring is empty")
        return self._rows.pop(0)

    def push(self, row, fence):
        self._rows.append(SpareRow(row, fence))


@dataclass
class MigrationTransaction:
    """One asynchronous plan from enqueue to commit."""

    plan: Any
    preflight: Preflight
    entries: list[tuple[int, Any, SpareRow, SpareRow]] = field(default_factory=list)
    transfer_event: Any = None
    state: str = "enqueued"
    enqueued_at: float = 0.0


class _DoneEvent:
    """A completed event for non-CUDA devices and tests."""

    def query(self):
        return True

    def synchronize(self):
        return None


class _NullStream:
    cuda_stream = -1

    def synchronize(self):
        return None


def _migration_stream(device):
    """The dedicated copy stream; one per process, created on first use."""
    import torch

    if device.type != "cuda":
        return _NullStream()
    stream = _STREAMS.get(device)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _STREAMS[device] = stream
    return stream


_STREAMS: dict[Any, Any] = {}


def _record_event(stream):
    """Record an event on `stream` after everything queued so far."""
    import torch

    if isinstance(stream, _NullStream) or not hasattr(stream, "record_event"):
        return _DoneEvent()
    event = torch.cuda.Event()
    event.record(stream)
    return event


def _stream_wait_event(stream, event):
    """Make `stream` wait for `event`; no-op for completed CPU events."""
    if event is None or isinstance(event, _DoneEvent):
        return
    if hasattr(stream, "wait_event"):
        stream.wait_event(event)


class _OnStream:
    def __init__(self, stream):
        self.stream = stream
        self._context = None

    def __enter__(self):
        import torch

        if not isinstance(self.stream, _NullStream):
            self._context = torch.cuda.stream(self.stream)
            self._context.__enter__()
        return self

    def __exit__(self, *exc):
        if self._context is not None:
            self._context.__exit__(*exc)
        return False


def _on_stream(stream):
    """Context manager: queue copies on `stream` (no-op off CUDA)."""
    return _OnStream(stream)
