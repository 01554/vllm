# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-level expert weight cache with the CachedWeightProvider surface.

The same public surface as CachedWeightProvider (`prepare`, `plan_chunks`,
`plan_expert_groups`, `invalidate`, `buf_*`, `capacity`, `split`, `hits`,
`misses`) plus per-expert global scales for NVFP4 (`buf_w13_scale_2`,
`buf_w2_scale_2`), an owner-stream contract, and counters.

Staged copies (this head), on CUDA:
1. At `prepare()` the previous forward's completion is recorded as a
   release event on the owner compute stream; the provider-owned copy
   stream waits on it before touching any slot.
2. Victims are chosen among rows not requested in this forward; every
   missing expert's six tensors are copied from the pinned host source
   into its slot on the copy stream.
3. A ready event is recorded on the copy stream and the owner compute
   stream waits on it, so the consumer's kernels are ordered after the
   copies without a host synchronization.
4. The forward map returned to the consumer is a separate object from
   the resident map and refers to exactly this generation of slots.
Pinned source rows and the slot assignments stay referenced by the
provider until the ready event has been waited on. On CPU the same steps
run synchronously. Calls may move between streams; the release event is
recorded on the previous caller's stream.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    ExpertWeightResult,
    MoECacheSplit,
    _pinned_cpu_copy,
)

logger = init_logger(__name__)
_STATS_LOG_INTERVAL = 1000


class RowCacheWeightProvider:
    def __init__(
        self,
        capacity: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        split: MoECacheSplit = "token",
        *,
        w13_scale_2: torch.Tensor | None = None,
        w2_scale_2: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.split: MoECacheSplit = split
        self._num_experts = w13_weight.size(0)
        self.hits = 0
        self.misses = 0
        self.promotions = 0
        self.evictions = 0
        self._prepare_calls = 0
        cache_device: torch.device
        if device is not None:
            cache_device = torch.device(device)
        elif w13_weight.device.type != "cpu":
            cache_device = w13_weight.device
        else:
            # CPU-resident source weights and no device given: fall back to
            # the current accelerator. Production passes the consumer's device
            # (an accelerator being compiled in does not prove a usable
            # driver); tests pass device="cpu".
            cache_device = torch.accelerator.current_accelerator() or torch.device(
                "cpu"
            )
        self.device = cache_device
        self._owner_stream: torch.cuda.Stream | None = None
        self.owner_changes = 0
        self._copy_stream: torch.cuda.Stream | None = None
        self._release_event: torch.cuda.Event | None = None
        self._ready_event: torch.cuda.Event | None = None
        self._generation = 0
        self._in_flight: list[
            tuple[int, int]
        ] = []  # (expert, slot) copied last prepare

        def host(t: torch.Tensor | None) -> torch.Tensor | None:
            # Source contract: on CUDA the source is pinned host memory; an
            # input that is already pinned is aliased (same helper as
            # CachedWeightProvider) and must stay immutable while the
            # provider lives; anything else is copied.
            if t is None:
                return None
            if cache_device.type == "cuda":
                return _pinned_cpu_copy(t)
            return t.detach().cpu().clone().contiguous()

        self._cpu = {
            "w13": host(w13_weight),
            "w2": host(w2_weight),
            "w13_scale": host(w13_scale),
            "w2_scale": host(w2_scale),
            "w13_scale_2": host(w13_scale_2),
            "w2_scale_2": host(w2_scale_2),
        }
        self._buf = {
            name: (
                None
                if src is None
                else torch.empty(
                    (capacity, *src.shape[1:]), dtype=src.dtype, device=cache_device
                )
            )
            for name, src in self._cpu.items()
        }
        # global expert id -> slot, -1 when not resident
        self._map = torch.full(
            (self._num_experts,), -1, dtype=torch.int32, device=cache_device
        )
        self._map_host = [-1] * self._num_experts
        self._lru: OrderedDict[int, int] = OrderedDict()  # expert -> slot, LRU order
        self._free = list(range(capacity - 1, -1, -1))

    # --- surface shared with CachedWeightProvider ---------------------------

    @property
    def buf_w13(self) -> torch.Tensor:
        return self._buf["w13"]

    @property
    def buf_w2(self) -> torch.Tensor:
        return self._buf["w2"]

    @property
    def buf_w13_scale(self) -> torch.Tensor | None:
        return self._buf["w13_scale"]

    @property
    def buf_w2_scale(self) -> torch.Tensor | None:
        return self._buf["w2_scale"]

    @property
    def buf_w13_scale_2(self) -> torch.Tensor | None:
        return self._buf["w13_scale_2"]

    @property
    def buf_w2_scale_2(self) -> torch.Tensor | None:
        return self._buf["w2_scale_2"]

    @property
    def expert_map(self) -> torch.Tensor:
        return self._map

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "promotions": self.promotions,
            "evictions": self.evictions,
            "resident": len(self._lru),
            "capacity": self.capacity,
            "prepare_calls": self._prepare_calls,
            "generation": self._generation,
            "last_copies": len(self._in_flight),
            "owner_changes": self.owner_changes,
            "slot_bytes": self.slot_bytes(),
            "host_bytes": self.host_bytes(),
        }

    def slot_bytes(self) -> int:
        """Device bytes held by the slot buffers (all six tensors)."""
        return sum(
            t.numel() * t.element_size() for t in self._buf.values() if t is not None
        )

    def host_bytes(self) -> int:
        """Pinned host bytes of the source (aliased when the input was pinned)."""
        return sum(
            t.numel() * t.element_size() for t in self._cpu.values() if t is not None
        )

    def invalidate(self, expert_id: int) -> None:
        # Does not take ownership: the reader ordering for a later reuse comes
        # from the next prepare(), which records its release event on the
        # stream of the last forward.
        if self.device.type == "cuda" and self._copy_stream is not None:
            # Wait for the copies that may still target this slot. This does
            # NOT wait for the previous reader on the owner stream: a freed
            # slot is only rewritten by a later prepare(), which records a new
            # release event on the owner stream before any copy, so the
            # previous reader is ordered ahead of the reuse by that contract.
            assert self._ready_event is not None
            self._copy_stream.synchronize()
        slot = self._lru.pop(expert_id, None)
        if slot is not None:
            self._free.append(slot)
            self._map_host[expert_id] = -1
            self._upload_map(self._map, self._map_host)

    @torch.compiler.disable
    def plan_chunks(self, topk_ids: torch.Tensor) -> list[tuple[slice, list[int]]]:
        ids = topk_ids.detach().cpu()
        plan: list[tuple[slice, list[int]]] = []
        start = 0
        seen: set[int] = set()
        rows = ids.shape[0]
        for r in range(rows):
            row_ids = {int(e) for e in ids[r].tolist() if e >= 0}
            if len(row_ids) > self.capacity:
                raise RuntimeError(
                    f"RowCacheWeightProvider: token {r} routes to "
                    f"{len(row_ids)} experts but capacity is {self.capacity}"
                )
            if len(seen | row_ids) > self.capacity:
                plan.append((slice(start, r), sorted(seen)))
                start, seen = r, set()
            seen |= row_ids
        plan.append((slice(start, rows), sorted(seen)))
        return plan

    @torch.compiler.disable
    def plan_expert_groups(self, topk_ids: torch.Tensor) -> list[list[int]]:
        unique = sorted(
            {int(e) for e in topk_ids.detach().cpu().reshape(-1).tolist() if e >= 0}
        )
        return [
            unique[i : i + self.capacity] for i in range(0, len(unique), self.capacity)
        ] or [[]]

    def prepare(
        self, topk_ids: torch.Tensor, unique_ids: list[int] | None = None
    ) -> ExpertWeightResult:
        """Make `unique_ids` resident and return the map selecting them.

        Validation (capacity, range, duplicates) happens before any state
        change, including the owner-stream record and the call counter.
        Stream contract: the release event is recorded on the stream of the
        previous accepted call (where that forward's kernels read the slots)
        and the ready event is awaited by the current stream, so calls may
        move between streams (profile run, graph capture, replay).
        """
        if unique_ids is None:
            unique_ids = sorted(
                {int(e) for e in topk_ids.detach().cpu().reshape(-1).tolist() if e >= 0}
            )
        if len(unique_ids) > self.capacity:
            raise RuntimeError(
                f"RowCacheWeightProvider: {len(unique_ids)} unique experts "
                f"requested but capacity is {self.capacity}"
            )
        needed = set(unique_ids)
        if len(needed) != len(unique_ids):
            raise ValueError(
                "RowCacheWeightProvider: duplicate expert ids in unique_ids"
            )
        bad = [e for e in unique_ids if not (0 <= e < self._num_experts)]
        if bad:
            raise ValueError(
                f"RowCacheWeightProvider: expert ids out of range: {bad[:4]} "
                f"(num_experts={self._num_experts})"
            )
        self._prepare_calls += 1
        self._generation += 1
        if self._prepare_calls % _STATS_LOG_INTERVAL == 0:
            st = self.stats()
            logger.info(
                "Row expert cache: %d hits, %d misses, %d evictions, "
                "%d/%d slots resident, slot %.1f MiB, host %.1f MiB",
                st["hits"],
                st["misses"],
                st["evictions"],
                len(self._lru),
                self.capacity,
                st["slot_bytes"] / 2**20,
                st["host_bytes"] / 2**20,
            )
        cuda = self.device.type == "cuda"
        if cuda:
            previous, owner = self._take_owner_stream()
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream(self.device)
                self._release_event = torch.cuda.Event()
                self._ready_event = torch.cuda.Event()
            assert self._release_event is not None and self._ready_event is not None
            # (1) the previous reader ran on the stream that was current at
            # the previous prepare(); it must finish before any slot is
            # rewritten: record there, wait on the copy stream. The consumer
            # of this call runs on the current stream, which may differ
            # (profile run, breakable-graph capture, replay).
            self._release_event.record(previous)
            self._copy_stream.wait_event(self._release_event)
        copies: list[tuple[int, int]] = []
        for e in unique_ids:
            if e in self._lru:
                self._lru.move_to_end(e)
                self.hits += 1
                continue
            self.misses += 1
            if self._free:
                slot = self._free.pop()
            else:
                # (2) victims are never rows requested in this forward
                victim, slot = next(
                    (k, v) for k, v in self._lru.items() if k not in needed
                )
                del self._lru[victim]
                self._map_host[victim] = -1
                self.evictions += 1
            copies.append((e, slot))
            self._lru[e] = slot
            self._map_host[e] = slot
            self.promotions += 1
        # One non-blocking upload per map. Per-element writes to a device
        # tensor (`map[e] = slot`) stage through pageable host memory and
        # block the host until the owner stream drains, i.e. until the
        # previous reader has finished -- which would hide the very ordering
        # the release event is meant to provide.
        self._upload_map(self._map, self._map_host)
        if cuda:
            assert self._copy_stream is not None and self._ready_event is not None
            with torch.cuda.stream(self._copy_stream):
                for e, slot in copies:
                    self._fill_slot(e, slot)
            # (3) consumer kernels on the owner stream wait for the copies
            self._ready_event.record(self._copy_stream)
            torch.cuda.current_stream(self.device).wait_event(self._ready_event)
        else:
            for e, slot in copies:
                self._fill_slot(e, slot)
        self._in_flight = copies
        # Rows not requested in this forward are hidden (-1) for this call.
        forward_host = [-1] * self._num_experts
        for e in unique_ids:
            forward_host[e] = self._map_host[e]
        forward_map = self._upload_map(None, forward_host)
        return ExpertWeightResult(
            w1=self._buf["w13"],
            w2=self._buf["w2"],
            expert_map=forward_map,
            w1_scale=self._buf["w13_scale"],
            w2_scale=self._buf["w2_scale"],
        )

    # --- internals ----------------------------------------------------------

    def _upload_map(
        self, target: torch.Tensor | None, values: list[int]
    ) -> torch.Tensor:
        """Copy `values` into `target` (or a new tensor) with one pinned upload.

        Per-element scalar writes to a device tensor stage through pageable
        memory and block the host until the stream drains; this path removes
        that staging (other host waits, e.g. allocation, are not claimed away).

        The staging tensor is pinned when the map lives on CUDA; the caching
        host allocator keeps it alive until the enqueued copy has consumed it.
        """
        host = torch.tensor(values, dtype=torch.int32)
        cuda = self._map.device.type == "cuda"
        if cuda:
            host = host.pin_memory()
        if target is None:
            return host.to(self._map.device, non_blocking=cuda)
        target.copy_(host, non_blocking=cuda)
        return target

    def _take_owner_stream(self) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
        """Return (previous owner, current stream) and make the current stream
        the owner. The owner is the stream the last accepted prepare() ran on,
        i.e. where that forward's kernels read the slots."""
        current = torch.cuda.current_stream(self.device)
        previous = self._owner_stream
        if previous is None:
            previous = current
        elif previous.cuda_stream != current.cuda_stream:
            self.owner_changes += 1
        self._owner_stream = current
        return previous, current

    def _fill_slot(self, expert: int, slot: int) -> None:
        for name, src in self._cpu.items():
            buf = self._buf[name]
            if src is None or buf is None:
                continue
            buf[slot].copy_(src[expert], non_blocking=self.device.type == "cuda")
