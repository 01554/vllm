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
run synchronously. Re-entry from a different stream is rejected.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    ExpertWeightResult,
    MoECacheSplit,
    _pinned_cpu_copy,
)


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
        self._owner_stream: int | None = None
        self._copy_stream: torch.cuda.Stream | None = None
        self._release_event: torch.cuda.Event | None = None
        self._ready_event: torch.cuda.Event | None = None
        self._generation = 0
        self._in_flight: list[
            tuple[int, int]
        ] = []  # (expert, slot) copied last prepare

        def host(t: torch.Tensor | None) -> torch.Tensor | None:
            # The provider owns its source: a clone, never an alias of the
            # caller's tensor (pinned when the cache is on CUDA).
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
        }

    def invalidate(self, expert_id: int) -> None:
        self._check_owner_stream()
        if self.device.type == "cuda" and self._copy_stream is not None:
            # Slot release must be ordered after the copies that targeted it
            # and after the previous reader on the owner stream.
            assert self._ready_event is not None
            torch.cuda.current_stream(self.device).wait_event(self._ready_event)
            self._copy_stream.synchronize()
        slot = self._lru.pop(expert_id, None)
        if slot is not None:
            self._free.append(slot)
            self._map_host[expert_id] = -1
            self._map[expert_id] = -1

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
        Owner-stream contract: the first accepted call records the current
        stream; a later call from a different stream is rejected.
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
        self._check_owner_stream()
        self._prepare_calls += 1
        self._generation += 1
        cuda = self.device.type == "cuda"
        if cuda:
            owner = torch.cuda.current_stream(self.device)
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream(self.device)
                self._release_event = torch.cuda.Event()
                self._ready_event = torch.cuda.Event()
            assert self._release_event is not None and self._ready_event is not None
            # (1) previous reader on the owner stream must finish before any
            # slot is rewritten: record on the owner stream, wait on the copy
            # stream.
            self._release_event.record(owner)
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
                self._map[victim] = -1
                self.evictions += 1
            copies.append((e, slot))
            self._lru[e] = slot
            self._map_host[e] = slot
            self._map[e] = slot
            self.promotions += 1
        if cuda:
            assert self._copy_stream is not None and self._ready_event is not None
            with torch.cuda.stream(self._copy_stream):
                for e, slot in copies:
                    self._fill_slot(e, slot)
            # (3) consumer kernels on the owner stream wait for the copies
            self._ready_event.record(self._copy_stream)
            torch.cuda.current_stream(self.device).wait_event(self._ready_event)
            self._in_flight = copies
        else:
            for e, slot in copies:
                self._fill_slot(e, slot)
        # Rows not requested in this forward are hidden (-1) for this call.
        forward_map = torch.full_like(self._map, -1)
        for e in unique_ids:
            forward_map[e] = self._map_host[e]
        return ExpertWeightResult(
            w1=self._buf["w13"],
            w2=self._buf["w2"],
            expert_map=forward_map,
            w1_scale=self._buf["w13_scale"],
            w2_scale=self._buf["w2_scale"],
        )

    # --- internals ----------------------------------------------------------

    def _check_owner_stream(self) -> None:
        if self.device.type != "cuda":
            return
        current = torch.cuda.current_stream(self.device).cuda_stream
        if self._owner_stream is None:
            self._owner_stream = current
        elif current != self._owner_stream:
            raise RuntimeError(
                "RowCacheWeightProvider: prepare() from a different stream "
                f"(owner {self._owner_stream:#x}, current {current:#x}) is not "
                "supported in this version"
            )

    def _fill_slot(self, expert: int, slot: int) -> None:
        for name, src in self._cpu.items():
            buf = self._buf[name]
            if src is None or buf is None:
                continue
            buf[slot].copy_(src[expert], non_blocking=self.device.type == "cuda")
