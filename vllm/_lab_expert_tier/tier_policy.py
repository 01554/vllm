# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-wide heat policy for compact, exclusive CPU/GPU expert banks.

Heat storage uses NumPy float64 arrays; NumPy is a direct vLLM common dependency.

Adapted from 01554/llama.cpp, expert-tier commit
7e6be0190af284b576690545c7cf38f3c9f2f453 (MIT; LICENSE.llama-cpp):
  https://github.com/01554/llama.cpp/blob/7e6be0190af284b576690545c7cf38f3c9f2f453/src/llama-expert-heatmap.cpp
    update_from_graph, decay_all, get_top_s
  https://github.com/01554/llama.cpp/blob/7e6be0190af284b576690545c7cf38f3c9f2f453/src/llama-expert-hotstore.cpp
    copy_top_s, resync_top_s, maybe_resync

Heat counts selections, decays once per model step (not once per token/layer),
and ranks prospective hot experts. Stable slots, the hysteresis/dwell gate,
integer cadence boundaries, prefill migration freeze, and the MODEL-GLOBAL
cumulative swap budget follow those functions. Budget is banked: this is not a
maximum burst size at each step. The original pre-increment comparison permits
ceil(K * tokens) swaps for fractional K; that behavior is retained. K=1 is the
initial runtime setting. One swap means an expert pair, across all its tensors.

New phase-two extension (not in the original C++): max_swaps_per_resync bounds
the total expert-pair swaps in one planned resync, across all layers. Zero means
unlimited and preserves the original behavior. A positive limit applies in
addition to the global cumulative budget; unspent credit is still banked. It
does not change heat ranking, layer order, cadence, or the compact swap operation.

Explicit adaptations: equal scores rank by ascending expert ID (C++ leaves
ties unspecified); validated real-token masks and zero-weight filtering remove
vLLM padding; initial_scores are already-scaled heat rather than the original
CSV prior's model-specific factor 6; state and budget belong to one model, not a
process global; Python float arithmetic replaces C++ float32. Defaults here are
the selected lab configuration, not claims about upstream constructor defaults.

Physical code must initialize banks from these initial maps, then, at the end
of a full model step, observe_step(), plan_resync(), execute plan.swaps IN ORDER,
and commit(plan), including an empty plan. For each swap and each tensor slice:
GPU hot -> RAM TEMP, CPU cold -> GPU hot, TEMP -> that vacated CPU cold slot.
No maps or swap counters publish until commit. Any physical-copy failure after
copying begins must poison/stop the engine; discard() is only for a plan whose
physical execution has not begun (or has been completely rolled back).
This policy owns no tensor, performs no DMA, and is not a demand-fill cache.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class Swap:
    layer: int
    hot_slot: int
    cold_slot: int
    old_expert: int
    new_expert: int


@dataclass(frozen=True)
class SwapPlan:
    swaps: tuple[Swap, ...]
    tokens_total: int
    base_version: int


@dataclass(frozen=True)
class _PlannedState:
    plan: SwapPlan
    hot: tuple[tuple[int, ...], ...]
    cold: tuple[tuple[int, ...], ...]
    dwell: tuple[tuple[int, ...], ...]


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(name: str, value: float, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return value


class TierPolicy:
    """One non-thread-safe policy shared by all MoE layers of one model.

    routes_by_layer and optional routing_weights_by_layer have shape
    [num_layers][padded_token_rows][routing_lanes]. The full model's routes must
    be submitted exactly once per step. valid_token_mask is one bool per row;
    num_tokens is its true count, or the number of rows when no mask is supplied.
    Original expert IDs 0..num_experts-1 are valid, including expert 0. Integer
    IDs outside that range are sentinels and ignored. Zero-weight lanes and
    masked rows add no heat; positive weights count once, regardless of size.
    Non-finite/negative weights on real rows are rejected, even for sentinel IDs.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        hot_slots: int,
        *,
        decay: float = 0.999,
        sync_period: int = 50,
        hysteresis: float = 1.3,
        dwell_tokens: int = 0,
        swaps_per_token: float = 1.0,
        max_swaps_per_resync: int = 0,
        initial_scores: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.num_layers = _integer("num_layers", num_layers, 1)
        self.num_experts = _integer("num_experts", num_experts, 1)
        self.hot_slots = _integer("hot_slots", hot_slots, 0)
        if self.hot_slots > self.num_experts:
            raise ValueError("hot_slots cannot exceed num_experts")
        self.decay = _finite("decay", decay)
        if self.decay > 1:
            raise ValueError("decay must be <= 1")
        self.sync_period = _integer("sync_period", sync_period, 0)
        self.hysteresis = _finite("hysteresis", hysteresis)
        self.dwell_tokens = _integer("dwell_tokens", dwell_tokens, 0)
        self.swaps_per_token = _finite("swaps_per_token", swaps_per_token)
        self.max_swaps_per_resync = _integer(
            "max_swaps_per_resync", max_swaps_per_resync, 0
        )
        if initial_scores is None:
            self._heat = np.zeros((self.num_layers, self.num_experts), dtype=np.float64)
        else:
            if len(initial_scores) != self.num_layers:
                raise ValueError("initial_scores must have num_layers rows")
            heat = []
            for row in initial_scores:
                if len(row) != self.num_experts:
                    raise ValueError("initial_scores must have num_experts columns")
                heat.append([_finite("initial score", score) for score in row])
            self._heat = np.asarray(heat, dtype=np.float64)

        self._expert_ids = np.arange(self.num_experts, dtype=np.intp)
        self._hot = tuple(tuple(self._top(layer)) for layer in range(self.num_layers))
        self._cold = tuple(
            tuple(e for e in range(self.num_experts) if e not in set(hot))
            for hot in self._hot
        )
        # copy_top_s initializes incumbents as already eligible for displacement.
        self._dwell = tuple((self.dwell_tokens,) * self.hot_slots for _ in self._hot)
        self._expert_to_hot, self._expert_to_cold = self._inverse_maps(
            self._hot, self._cold
        )
        self.tokens_total = 0
        self.swaps_total = 0
        self.last_sync_tokens = 0
        self.version = 0  # LUT version, incremented only by a nonempty committed plan.
        self._last_step_tokens = 0
        self._pending: _PlannedState | None = None
        # Device snapshots carry absolute heat and token totals.  These guards
        # belong to the import protocol, while version, cadence, maps, and
        # dwell remain policy-owned state.
        self._device_session: str | int | None = None
        self._device_session_set = False
        self._device_sequence = -1
        self._device_forwards = 0
        self._device_due = False

    @property
    def hot_to_expert(self) -> tuple[tuple[int, ...], ...]:
        return self._hot

    @property
    def cold_to_expert(self) -> tuple[tuple[int, ...], ...]:
        return self._cold

    @property
    def expert_to_hot(self) -> tuple[tuple[int, ...], ...]:
        """Original expert -> real hot slot, or -1; no padding slots here."""
        return self._expert_to_hot

    @property
    def expert_to_cold(self) -> tuple[tuple[int, ...], ...]:
        """Original expert -> compact cold slot, or -1."""
        return self._expert_to_cold

    @property
    def heat(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(row) for row in self._heat.tolist())

    @property
    def dwell_counts(self) -> tuple[tuple[int, ...], ...]:
        return self._dwell

    def _top(self, layer: int) -> list[int]:
        if not self.hot_slots:
            return []
        heat = self._heat[layer]
        if np.isnan(heat).any():
            # Validated updates should never create NaN, but retain this
            # defensive path because NumPy and Python order NaN differently;
            # the scalar sort preserves the historical Python ordering.
            values = heat.tolist()
            return sorted(range(self.num_experts), key=lambda e: (-values[e], e))[
                : self.hot_slots
            ]
        return np.lexsort((self._expert_ids, -heat))[: self.hot_slots].tolist()

    def _inverse_maps(self, hot, cold):
        all_hot, all_cold = [], []
        for hot_row, cold_row in zip(hot, cold):
            hot_map, cold_map = [-1] * self.num_experts, [-1] * self.num_experts
            for slot, expert in enumerate(hot_row):
                hot_map[expert] = slot
            for slot, expert in enumerate(cold_row):
                cold_map[expert] = slot
            all_hot.append(tuple(hot_map))
            all_cold.append(tuple(cold_map))
        return tuple(all_hot), tuple(all_cold)

    def observe_step(
        self,
        routes_by_layer: Sequence[Sequence[Sequence[int]]],
        num_tokens: int,
        routing_weights_by_layer: Sequence[Sequence[Sequence[float]]] | None = None,
        valid_token_mask: Sequence[bool] | None = None,
    ) -> None:
        """Atomically validate and collect one complete model step's routing.

        Multi-token steps collect heat/credit but freeze migration. A zero-token
        dummy step neither decays heat nor earns credit and cannot trigger sync.
        The caller must supply the mask for padded/dummy rows: zero weights alone
        cannot reveal whether an otherwise valid token should earn token credit.
        """
        if self._pending is not None:
            raise RuntimeError(
                "commit or discard the outstanding plan before observing"
            )
        num_tokens = _integer("num_tokens", num_tokens, 0)
        if len(routes_by_layer) != self.num_layers:
            raise ValueError(
                "routes_by_layer must contain every model layer exactly once"
            )
        rows = len(routes_by_layer[0])
        if valid_token_mask is None:
            if rows != num_tokens:
                raise ValueError(
                    "without valid_token_mask, row count must equal num_tokens"
                )
            mask = (True,) * rows
        else:
            mask = tuple(valid_token_mask)
            if len(mask) != rows or any(type(x) is not bool for x in mask):
                raise ValueError("valid_token_mask must contain one bool per row")
            if sum(mask) != num_tokens:
                raise ValueError("num_tokens must equal valid_token_mask true count")
        if (
            routing_weights_by_layer is not None
            and len(routing_weights_by_layer) != self.num_layers
        ):
            raise ValueError("routing weights must contain every model layer")

        # Validate the full step before modifying heat, so malformed later layers
        # cannot partially count a model token. Retain only compact per-expert counts.
        counts: list[dict[int, int]] = []
        for layer, route_rows in enumerate(routes_by_layer):
            if len(route_rows) != rows:
                raise ValueError("all layers must have the same token rows")
            weight_rows = (
                None
                if routing_weights_by_layer is None
                else routing_weights_by_layer[layer]
            )
            if weight_rows is not None and len(weight_rows) != rows:
                raise ValueError(
                    "routing weights and routes must have identical shapes"
                )
            layer_counts: dict[int, int] = {}
            for token, ids in enumerate(route_rows):
                weights = None if weight_rows is None else weight_rows[token]
                if weights is not None and len(weights) != len(ids):
                    raise ValueError(
                        "routing weights and routes must have identical shapes"
                    )
                if not mask[token]:
                    continue
                for lane, expert in enumerate(ids):
                    if isinstance(expert, bool) or not isinstance(expert, Integral):
                        raise ValueError("routing expert IDs must be integers")
                    weight = (
                        1.0
                        if weights is None
                        else _finite("routing weight", weights[lane])
                    )
                    if 0 <= expert < self.num_experts and weight > 0:
                        expert = int(expert)
                        layer_counts[expert] = layer_counts.get(expert, 0) + 1
            counts.append(layer_counts)
        if num_tokens:
            # NumPy's float64 multiply has the same IEEE-754 result as Python's
            # float multiply, while avoiding one Python loop over every expert.
            # Counts are still added below in their original layer/dict order.
            with np.errstate(all="ignore"):
                self._heat *= self.decay
                for layer, layer_counts in enumerate(counts):
                    heat = self._heat[layer]
                    for expert, count in layer_counts.items():
                        heat[expert] += count
            self.tokens_total += num_tokens  # once for the model, never per layer.
        self._last_step_tokens = num_tokens

    @staticmethod
    def _snapshot_value(snapshot, *names, default=None):
        for name in names:
            if hasattr(snapshot, name):
                return getattr(snapshot, name)
        return default

    def import_device_snapshot(self, snapshot) -> None:
        """Import absolute device heat exactly once for a snapshot sequence.

        The device has already applied every decay and integer count update in
        the snapshot.  Import therefore replaces ``_heat`` and token totals;
        it never calls :meth:`observe_step`, advances dwell, publishes maps,
        or changes policy version/cadence state.  A device observer should
        provide the base/session/sequence metadata so stale and duplicate
        snapshots fail before any policy state changes.

        ``last_step_tokens`` is part of the metadata because a multi-token
        prefill can cross a cadence boundary without being eligible to plan.
        The following single-token step must retain that opportunity even when
        it arrives in a later snapshot.
        """
        if self._pending is not None:
            raise RuntimeError(
                "commit or discard the outstanding plan before importing"
            )

        verified = self._snapshot_value(snapshot, "verified", default=True)
        error = self._snapshot_value(snapshot, "error", default=False)
        if verified is not True or error is True:
            raise RuntimeError(
                "device snapshot validation failed; delayed validation is not "
                "the legacy fail-closed path"
            )

        heat_value = self._snapshot_value(snapshot, "heat")
        if heat_value is None:
            raise ValueError("device snapshot must contain heat")
        try:
            heat = np.array(heat_value, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("device snapshot heat is not a numeric matrix") from exc
        if heat.shape != (self.num_layers, self.num_experts):
            raise ValueError("device snapshot heat shape does not match the policy")
        if not np.isfinite(heat).all() or (heat < 0).any():
            raise ValueError("device snapshot heat must be finite and nonnegative")

        tokens_value = self._snapshot_value(
            snapshot, "tokens_total", "tokens", default=None
        )
        if tokens_value is None:
            raise ValueError("device snapshot must contain an absolute token total")
        tokens_total = _integer("snapshot tokens_total", tokens_value, 0)

        last_step_value = self._snapshot_value(
            snapshot, "last_step_tokens", default=None
        )
        if last_step_value is None:
            # Compatibility with the first observer seam.  Full device
            # snapshots must carry this field to distinguish prefill and a
            # one-token boundary; a legacy snapshot is treated as one step.
            last_step_tokens = 1 if tokens_total > self.tokens_total else 0
        else:
            last_step_tokens = _integer(
                "snapshot last_step_tokens", last_step_value, 0
            )
        if last_step_tokens > tokens_total:
            raise ValueError(
                "device snapshot last_step_tokens exceeds its token total"
            )

        base_names = (
            "base_tokens_total",
            "base_version",
            "base_last_sync_tokens",
        )
        base_values = [
            self._snapshot_value(snapshot, name, default=None) for name in base_names
        ]
        has_base = any(value is not None for value in base_values)
        if has_base and any(value is None for value in base_values):
            raise ValueError("device snapshot base metadata must be complete")
        if has_base:
            base_tokens, base_version, base_last_sync = (
                _integer(name, value, 0)
                for name, value in zip(base_names, base_values)
            )
            if (base_tokens, base_version, base_last_sync) != (
                self.tokens_total,
                self.version,
                self.last_sync_tokens,
            ):
                raise RuntimeError("device snapshot base is stale")
            if base_tokens > tokens_total or base_last_sync > tokens_total:
                raise ValueError("device snapshot base exceeds its token total")

        session = self._snapshot_value(snapshot, "session_id", default=None)
        sequence_value = self._snapshot_value(snapshot, "sequence", default=None)
        has_sequence = session is not None or sequence_value is not None
        if has_sequence and (session is None or sequence_value is None):
            raise ValueError("device snapshot session and sequence must be complete")
        if has_sequence:
            if not isinstance(session, (str, int)) or isinstance(session, bool):
                raise ValueError(
                    "device snapshot session_id must be a string or integer"
                )
            sequence = _integer("snapshot sequence", sequence_value, 0)
            if self._device_session_set and session != self._device_session:
                raise RuntimeError("device snapshot belongs to a stale session")
            if self._device_sequence >= 0 and sequence <= self._device_sequence:
                raise RuntimeError("device snapshot is duplicate or out of order")
        else:
            sequence = None

        forwards_value = self._snapshot_value(
            snapshot, "forwards_total", "forwards", default=None
        )
        if forwards_value is not None:
            forwards = _integer("snapshot forwards", forwards_value, 0)
        else:
            forwards = None
        if has_sequence and forwards is not None and forwards <= self._device_forwards:
            raise RuntimeError("device snapshot forwards are duplicate or out of order")

        if tokens_total < self.tokens_total:
            raise RuntimeError("device snapshot token total is stale")
        if (
            tokens_total == self.tokens_total
            and (
                not has_sequence
                or forwards is None
                or forwards <= self._device_forwards
            )
        ):
            raise RuntimeError("device snapshot contains no new observation")

        due_value = self._snapshot_value(snapshot, "resync_due", default=None)
        if due_value is not None and type(due_value) is not bool:
            raise ValueError("device snapshot resync_due must be a bool")
        due = bool(due_value) if due_value is not None else False

        # All validation above is complete before replacing any state.  The
        # copy is intentional: the policy must not alias a snapshot's storage.
        self._heat = heat
        self.tokens_total = tokens_total
        self._last_step_tokens = last_step_tokens
        if has_sequence:
            self._device_session = session
            self._device_session_set = True
            self._device_sequence = sequence
            if forwards is not None:
                self._device_forwards = forwards
        self._device_due = self._device_due or due

    def import_snapshot(self, snapshot) -> None:
        """Compatibility alias used by the runtime observer seam."""
        self.import_device_snapshot(snapshot)

    def plan_resync(self) -> SwapPlan | None:
        """Return a cadence-gated transaction; published maps stay unchanged.

        Repeated calls before commit return the same plan. A due plan can contain
        zero swaps; it must still be committed to age dwell and advance cadence.
        A positive max_swaps_per_resync limits this plan, not each layer. The
        original global budget can impose a smaller limit; neither cap spends
        credit before commit or prevents dwell aging in later layers.
        """
        if self._pending is not None:
            return self._pending.plan
        if self._last_step_tokens != 1 or self.sync_period <= 0 or self.hot_slots <= 0:
            return None
        if (
            self.tokens_total // self.sync_period
            <= self.last_sync_tokens // self.sync_period
            and not self._device_due
        ):
            return None

        elapsed = self.tokens_total - self.last_sync_tokens
        hot, cold = [list(row) for row in self._hot], [list(row) for row in self._cold]
        dwell = [list(row) for row in self._dwell]
        swaps: list[Swap] = []
        for layer in range(self.num_layers):
            # Snapshot membership matches resync_top_s. Simulated live slots/maps
            # still change after EACH swap, including repeated use of a hot slot.
            resident_set = set(hot[layer])
            cold_slot_of = {expert: slot for slot, expert in enumerate(cold[layer])}
            for new_expert in self._top(layer):
                if new_expert in resident_set:
                    continue
                if (
                    self.swaps_total + len(swaps)
                    >= self.swaps_per_token * self.tokens_total
                ):
                    break
                if (
                    self.max_swaps_per_resync
                    and len(swaps) >= self.max_swaps_per_resync
                ):
                    break
                eligible = [
                    slot
                    for slot, old in enumerate(hot[layer])
                    if self.hysteresis <= 0
                    or (
                        dwell[layer][slot] >= self.dwell_tokens
                        and float(self._heat[layer][new_expert])
                        >= self.hysteresis * float(self._heat[layer][old])
                    )
                ]
                if not eligible:
                    break
                # C++ retains the first hot slot on equal incumbent heat.
                hot_slot = min(
                    eligible,
                    key=lambda slot: float(self._heat[layer][hot[layer][slot]]),
                )
                cold_slot = cold_slot_of.get(new_expert)
                if cold_slot is None:
                    continue
                old_expert = hot[layer][hot_slot]
                swaps.append(Swap(layer, hot_slot, cold_slot, old_expert, new_expert))
                hot[layer][hot_slot] = new_expert
                cold[layer][cold_slot] = old_expert
                del cold_slot_of[new_expert]
                cold_slot_of[old_expert] = cold_slot
                dwell[layer][hot_slot] = -elapsed
            # Match original order: test dwell BEFORE adding elapsed tokens.
            dwell[layer] = [age + elapsed for age in dwell[layer]]

        plan = SwapPlan(tuple(swaps), self.tokens_total, self.version)
        self._pending = _PlannedState(
            plan,
            tuple(map(tuple, hot)),
            tuple(map(tuple, cold)),
            tuple(map(tuple, dwell)),
        )
        return plan

    def _require_pending(self, plan: SwapPlan) -> _PlannedState:
        if self._pending is None or self._pending.plan is not plan:
            raise ValueError(
                "plan is not this policy's current outstanding transaction"
            )
        return self._pending

    def commit(self, plan: SwapPlan) -> None:
        """Publish after all ordered physical swaps and tensor copies succeed."""
        pending = self._require_pending(plan)
        inverse_hot, inverse_cold = self._inverse_maps(pending.hot, pending.cold)
        self._hot, self._cold, self._dwell = pending.hot, pending.cold, pending.dwell
        self._expert_to_hot, self._expert_to_cold = inverse_hot, inverse_cold
        self.swaps_total += len(plan.swaps)
        self.last_sync_tokens = plan.tokens_total
        if plan.swaps:
            self.version += 1
        self._device_due = False
        self._pending = None

    def discard(self, plan: SwapPlan) -> None:
        """Forget an unexecuted plan; does not roll back any physical copies."""
        self._require_pending(plan)
        self._pending = None
