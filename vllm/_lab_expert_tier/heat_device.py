# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-resident expert heat observation.

The accumulator in this module deliberately has no dependency on the runtime
coordinator.  It can therefore be used by a CUDA-graph observer and by the
CPU differential tests without importing the model-runner integration.

Routing records are reduced into an int64 count matrix.  A model step then
performs two separate float64 operations on the persistent heat tensor:
``heat *= decay`` followed by ``heat += counts``.  The integer reduction is
important for duplicate expert IDs and avoids rounding differences from a
floating ``scatter_add``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar

import torch


def _integer(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(name: str, value: float, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class DeviceSnapshot:
    """An immutable host snapshot of device heat and observation metadata.

    ``tokens``, ``forwards``, and ``route_total`` are cumulative counters for
    the observer session.  The explicit ``*_total`` fields repeat those
    absolute values so a policy importer does not have to infer which seam it
    received.  The runtime coordinator differences cumulative counters against
    its per-session reporting cursor.  The ``base_*`` fields identify the
    policy state against which the accumulator was started.  They are optional
    for compatibility with the first observer seam, but a production device
    observer should always populate them.

    The heat value is stored as nested tuples, so a later device update cannot
    mutate a previously returned snapshot.

    ``route_hot_available`` is false when any recorded layer omits its device
    hot-map.  In that case ``route_hot=0`` is a compatibility placeholder, not
    a measured zero hit rate.
    """

    kind: ClassVar[str] = "device_snapshot"

    heat: tuple[tuple[float, ...], ...]
    tokens: int
    forwards: int
    route_hot: int = 0
    route_total: int = 0
    verified: bool = True
    last_step_tokens: int | None = None
    base_tokens_total: int | None = None
    base_version: int | None = None
    base_last_sync_tokens: int | None = None
    session_id: str | int | None = None
    sequence: int | None = None
    resync_due: bool | None = None
    error: bool = False
    tokens_total: int | None = None
    forwards_total: int | None = None
    route_total_total: int | None = None
    route_hot_available: bool = False

    def __post_init__(self) -> None:
        # Older callers supplied an absolute ``tokens`` value and did not know
        # about the explicit absolute field.  Retain that construction shape;
        # DeviceObserver always supplies both values.
        if self.tokens_total is None:
            object.__setattr__(self, "tokens_total", self.tokens)
        if self.forwards_total is None:
            object.__setattr__(self, "forwards_total", self.forwards)
        if self.route_total_total is None:
            object.__setattr__(self, "route_total_total", self.route_total)

    @property
    def absolute_tokens(self) -> int:
        """Explicit alias for the absolute token total."""
        assert self.tokens_total is not None
        return self.tokens_total

    @property
    def absolute_forwards(self) -> int:
        """Explicit alias for the absolute forward total."""
        assert self.forwards_total is not None
        return self.forwards_total

    @property
    def observed_steps(self) -> int:
        """Alias for the number of model boundaries represented."""
        return self.forwards


# The longer name is useful to callers that want to distinguish this object
# from the runtime's result wrapper.  Both names intentionally denote the same
# immutable contract.
DeviceHeatSnapshot = DeviceSnapshot


@dataclass(frozen=True, slots=True)
class Deferred:
    """No host readback is needed for this model boundary."""

    kind: ClassVar[str] = "deferred"
    forwards: int = 1


@dataclass(frozen=True, slots=True)
class LegacyRoutes:
    """Host routes returned by the legacy observer seam."""

    kind: ClassVar[str] = "legacy_routes"
    routes: Any
    activity: Any
    mask: Any
    tokens: int


class DeviceHeatAccumulator:
    """Graph-capturable, fixed-shape device heat accumulator.

    ``record_layer`` may be called from a captured forward.  It performs no
    scalar device reads and makes no CPU copy.  ``finish_step`` is called at a
    model boundary with the runner's known true-token count.  Only
    ``snapshot``/``flush`` copy heat to the host.

    Args:
        num_layers: Number of MoE layers in the complete model step.
        num_experts: Number of experts in every layer.
        decay: Per-model-step float64 heat decay.
        sync_period: Token cadence used to mark a resync opportunity.
        initial_scores: Optional nonnegative float64 heat matrix.
        device: Device on which persistent tensors are allocated.
        top_k: Optional fixed routing width.  Once a record is accepted, its
            width is fixed even when ``top_k`` is omitted.
        max_rows: Optional fixed row capacity.
        enabled: Initial heat gate.  ``DeviceObserver`` starts disabled and
            opens it only at the coordinator's named heat-enable boundary.
        base_tokens_total: Policy token total when this accumulator starts.
        base_version: Policy LUT version when this accumulator starts.
        base_last_sync_tokens: Policy cadence cursor when this accumulator
            starts.
        session_id: Stable observer session identifier for snapshot guards.
    """

    _ID_DTYPES = (torch.int32, torch.int64)
    _HOT_MAP_DTYPES = (torch.int32, torch.int64)
    _FLAG_DTYPES = (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        *,
        decay: float = 0.999,
        sync_period: int = 50,
        initial_scores: Sequence[Sequence[float]] | torch.Tensor | None = None,
        device: torch.device | str | None = None,
        top_k: int | None = None,
        max_rows: int | None = None,
        enabled: bool = True,
        base_tokens_total: int = 0,
        base_version: int = 0,
        base_last_sync_tokens: int = 0,
        session_id: str | int | None = None,
    ) -> None:
        self.num_layers = _integer("num_layers", num_layers, 1)
        self.num_experts = _integer("num_experts", num_experts, 1)
        self.decay = _number("decay", decay)
        if self.decay > 1.0:
            raise ValueError("decay must be <= 1")
        self.sync_period = _integer("sync_period", sync_period, 0)
        self.top_k = None if top_k is None else _integer("top_k", top_k, 1)
        self.max_rows = None if max_rows is None else _integer("max_rows", max_rows, 1)
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a bool")
        self._base_tokens_total = _integer("base_tokens_total", base_tokens_total, 0)
        self._base_version = _integer("base_version", base_version, 0)
        self._base_last_sync_tokens = _integer(
            "base_last_sync_tokens", base_last_sync_tokens, 0
        )
        self.session_id = id(self) if session_id is None else session_id

        self.device = torch.device("cpu" if device is None else device)
        self.heat = torch.zeros(
            (self.num_layers, self.num_experts),
            dtype=torch.float64,
            device=self.device,
        )
        if initial_scores is not None:
            scores = torch.as_tensor(
                initial_scores, dtype=torch.float64, device=self.device
            )
            if tuple(scores.shape) != (self.num_layers, self.num_experts):
                raise ValueError(
                    "initial_scores must have shape [num_layers, num_experts]"
                )
            if not bool(torch.isfinite(scores).all().item()) or bool(
                (scores < 0).any().item()
            ):
                raise ValueError("initial_scores must be finite and nonnegative")
            self.heat.copy_(scores)

        self.counts = torch.zeros(
            (self.num_layers, self.num_experts),
            dtype=torch.int64,
            device=self.device,
        )
        self._scratch = torch.zeros_like(self.heat)
        self._expected_layer = torch.zeros((), dtype=torch.int64, device=self.device)
        self._error_flag = torch.zeros((), dtype=torch.bool, device=self.device)
        self._step_route_total = torch.zeros((), dtype=torch.int64, device=self.device)
        self._step_route_hot = torch.zeros((), dtype=torch.int64, device=self.device)
        self._route_total = torch.zeros((), dtype=torch.int64, device=self.device)
        self._route_hot = torch.zeros((), dtype=torch.int64, device=self.device)
        # These flags are device state because record_layer may execute only
        # during graph capture.  The captured zero/fill operations replay even
        # when Python does not call record_layer again.
        self._step_hot_map_seen = torch.zeros(
            (self.num_layers,), dtype=torch.bool, device=self.device
        )
        self._step_hot_map_missing = torch.zeros(
            (), dtype=torch.bool, device=self.device
        )
        self._hot_stats_initialized = torch.zeros(
            (), dtype=torch.bool, device=self.device
        )
        self._hot_stats_valid = torch.ones((), dtype=torch.bool, device=self.device)

        self._tokens_total = torch.tensor(
            self._base_tokens_total, dtype=torch.int64, device=self.device
        )
        self._last_sync_tokens = torch.tensor(
            self._base_last_sync_tokens, dtype=torch.int64, device=self.device
        )
        self._last_step_tokens = torch.zeros((), dtype=torch.int64, device=self.device)
        self._forward_count = torch.zeros((), dtype=torch.int64, device=self.device)
        self._step_tokens = torch.zeros((), dtype=torch.int64, device=self.device)
        self._step_valid_tokens = torch.zeros((), dtype=torch.int64, device=self.device)

        # These scalar tensors are inputs to the captured update.  The Python
        # heat-enable call changes their values at a named lifecycle boundary,
        # so a graph captured while disabled still observes later enabling.
        self._enabled = torch.tensor(enabled, dtype=torch.bool, device=self.device)
        self._enabled_int = torch.tensor(
            int(enabled), dtype=torch.int64, device=self.device
        )
        self._enabled_float = torch.tensor(
            float(enabled), dtype=torch.float64, device=self.device
        )
        self._step_positive_float = torch.zeros(
            (), dtype=torch.float64, device=self.device
        )
        self._effective_float = torch.zeros((), dtype=torch.float64, device=self.device)
        self._decay_value = torch.tensor(
            self.decay, dtype=torch.float64, device=self.device
        )
        self._one = torch.ones((), dtype=torch.float64, device=self.device)
        self._zero = torch.zeros((), dtype=torch.float64, device=self.device)
        self._scale = torch.ones((), dtype=torch.float64, device=self.device)
        if enabled:
            self._scale.fill_(self.decay)
        self._resync_due = torch.zeros((), dtype=torch.bool, device=self.device)

        # Host metadata is derived from runner-supplied counts, never from a
        # per-step scalar device read.  It is used only to decide when a host
        # snapshot is allowed.
        self._host_enabled = enabled
        self._host_tokens_total = self._base_tokens_total
        self._host_last_sync_tokens = self._base_last_sync_tokens
        self._host_last_step_tokens = 0
        self._host_forwards = 0
        self._host_route_total = 0
        self._host_due = False
        self._pending_snapshot: DeviceSnapshot | None = None
        self._last_snapshot_forwards = 0
        self._last_snapshot_tokens = self._base_tokens_total
        self._last_snapshot_route_total = 0
        self._last_snapshot_route_hot = 0
        self._last_snapshot_step_tokens = 0
        self._sequence = -1

    @property
    def error_flag(self) -> torch.Tensor:
        """Persistent device error flag; it is not silently cleared per step."""
        return self._error_flag

    @property
    def error(self) -> bool:
        """Read the persistent error at an explicit host boundary."""
        return bool(self._error_flag.detach().cpu().item())

    @property
    def enabled(self) -> bool:
        """Current host view of the named heat-enable gate."""
        return self._host_enabled

    @property
    def tokens_total(self) -> int:
        """Known token total from runner metadata, without a device read."""
        return self._host_tokens_total

    @property
    def last_step_tokens(self) -> int:
        """Known true-token count of the latest enabled model boundary."""
        return self._host_last_step_tokens

    @property
    def forwards(self) -> int:
        """Number of enabled model boundaries represented by this accumulator."""
        return self._host_forwards

    @property
    def resync_due(self) -> bool:
        """Whether a cadence boundary has been crossed since the last sync."""
        return self._host_due

    @property
    def base_tokens_total(self) -> int:
        return self._base_tokens_total

    @property
    def base_version(self) -> int:
        return self._base_version

    @property
    def base_last_sync_tokens(self) -> int:
        return self._base_last_sync_tokens

    @property
    def route_total(self) -> int:
        """Cumulative selected-route count, read at an explicit boundary."""
        return int(self._route_total.detach().cpu().item())

    @property
    def route_hot(self) -> int:
        """Cumulative hot-route count, read at an explicit boundary."""
        return int(self._route_hot.detach().cpu().item())

    @property
    def route_hot_available(self) -> bool:
        """Whether the cumulative hot-route count is complete and measured."""
        available = self._hot_stats_initialized & self._hot_stats_valid
        return bool(available.detach().cpu().item())

    def _record_hot_map_state(self, layer_index: int, hot_map: Any) -> None:
        """Record map availability using replay-safe device operations."""
        if layer_index == 0:
            self._step_hot_map_seen.zero_()
            self._step_hot_map_missing.zero_()
        if hot_map is None:
            self._step_hot_map_missing.fill_(True)
        else:
            self._step_hot_map_seen[layer_index].fill_(True)

    def _set_gate(self, enabled: bool) -> None:
        """Update the persistent device gate at a named lifecycle boundary."""
        if not isinstance(enabled, bool):
            raise ValueError("heat_enabled must be a bool")
        self._host_enabled = enabled
        self._enabled.fill_(enabled)
        self._enabled_int.fill_(int(enabled))
        self._enabled_float.fill_(float(enabled))
        self._scale.fill_(self.decay if enabled else 1.0)

    def on_heat_enabled(self) -> None:
        """Open the persistent device gate after startup/warmup succeeds."""
        self._set_gate(True)

    def set_heat_enabled(self, enabled: bool) -> None:
        """Set the persistent gate at an explicit coordinator boundary."""
        self._set_gate(enabled)

    def clear_error(self) -> None:
        """Clear the error flag only at an explicit recovery boundary."""
        self._error_flag.zero_()

    def _normalise_record_args(
        self,
        ids: Any,
        activity_flags: Any,
        valid_mask: Any,
        rows: Any,
        hot_map: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Any]:
        # The runtime seam historically used (layer, rows, ids, active, valid),
        # while the device contract uses (layer, ids, active, valid, rows).
        # Accept both layouts so the observer can be wired independently.
        if isinstance(ids, Integral) and isinstance(rows, torch.Tensor):
            legacy_rows = int(ids)
            ids, activity_flags, valid_mask, rows = (
                activity_flags,
                valid_mask,
                rows,
                legacy_rows,
            )
        if not isinstance(ids, torch.Tensor):
            raise TypeError("ids must be a torch.Tensor")
        if not isinstance(activity_flags, torch.Tensor):
            raise TypeError("activity_flags must be a torch.Tensor")
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("valid_mask must be a torch.Tensor")
        rows = _integer("rows", rows, 0)
        return ids, activity_flags, valid_mask, rows, hot_map

    def record_layer(
        self,
        layer_index: int,
        ids: Any,
        activity_flags: Any = None,
        valid_mask: Any = None,
        rows: Any = None,
        hot_map: Any = None,
    ) -> None:
        """Reduce one fixed-shape layer record into integer device counts.

        ``ids`` has shape ``[capacity, top_k]`` and the activity and valid
        tensors have matching leading dimensions.  Invalid IDs on valid rows
        set the persistent error flag; clamped safe indices ensure that the
        delayed snapshot validation cannot cause an out-of-bounds write.
        """
        layer_index = _integer("layer_index", layer_index, 0)
        if layer_index >= self.num_layers:
            raise ValueError("layer_index is outside the allocated layer count")
        ids, activity_flags, valid_mask, rows, hot_map = self._normalise_record_args(
            ids, activity_flags, valid_mask, rows, hot_map
        )
        if (
            rows > ids.shape[0]
            or rows > activity_flags.shape[0]
            or rows > valid_mask.shape[0]
        ):
            raise ValueError("rows exceeds the record tensor capacity")
        if ids.ndim != 2 or activity_flags.ndim != 2 or valid_mask.ndim != 1:
            raise ValueError(
                "routing records must have shapes [rows, top_k], [rows, top_k], [rows]"
            )
        if ids.shape != activity_flags.shape or ids.shape[0] != valid_mask.shape[0]:
            raise ValueError("routing record tensors have incompatible shapes")
        if (
            ids.device != self.device
            or activity_flags.device != self.device
            or valid_mask.device != self.device
        ):
            raise ValueError("routing record tensors must share the accumulator device")
        if ids.dtype not in self._ID_DTYPES:
            raise TypeError("ids must use int32 or int64")
        if activity_flags.dtype not in self._FLAG_DTYPES:
            raise TypeError("activity_flags must use a boolean or integer dtype")
        if valid_mask.dtype is not torch.bool:
            raise TypeError("valid_mask must use torch.bool")
        if self.top_k is not None and ids.shape[1] != self.top_k:
            raise ValueError("routing width does not match the fixed top_k")
        if self.max_rows is not None and rows > self.max_rows:
            raise ValueError("rows exceeds the fixed max_rows capacity")
        if hot_map is not None:
            if not isinstance(hot_map, torch.Tensor):
                raise TypeError("hot_map must be a torch.Tensor")
            if hot_map.ndim != 1 or hot_map.shape[0] != self.num_experts:
                raise ValueError("hot_map must have shape [num_experts]")
            if hot_map.device != self.device:
                raise ValueError("hot_map must share the accumulator device")
            if hot_map.dtype not in self._HOT_MAP_DTYPES:
                raise TypeError("hot_map must use int32 or int64")

        if layer_index == 0:
            # A graph capture can finish with a complete layer sequence but no
            # host ``finish_step``.  Treat that complete sequence as the
            # initial state of the next replay; retain errors only when the
            # preceding sequence was incomplete.
            self._error_flag.logical_or_(
                self._expected_layer.ne(0) & self._expected_layer.ne(self.num_layers)
            )
            self.counts.zero_()
            self._step_route_total.zero_()
            self._step_route_hot.zero_()
            self._step_valid_tokens.zero_()
            self._expected_layer.zero_()
        self._error_flag.logical_or_(self._expected_layer.ne(layer_index))
        self._record_hot_map_state(layer_index, hot_map)

        ids_view = ids[:rows].to(dtype=torch.int64)
        active_view = activity_flags[:rows].ne(0)
        valid_view = valid_mask[:rows]
        valid_count = valid_view.to(dtype=torch.int64).sum()
        if layer_index == 0:
            self._step_valid_tokens.copy_(valid_count)
        else:
            self._error_flag.logical_or_(valid_count.ne(self._step_valid_tokens))
        in_range = (ids_view >= 0) & (ids_view < self.num_experts)
        invalid_real = (~in_range) & valid_view[:, None]
        self._error_flag.logical_or_(invalid_real.any())

        # Clamp before flattening so invalid device values can never index out
        # of range.  Invalid entries are excluded from the integer reduction.
        safe_ids = ids_view.clamp(0, self.num_experts - 1)
        selected = active_view & valid_view[:, None] & in_range
        self.counts[layer_index].view(-1).index_add_(
            0,
            safe_ids.reshape(-1),
            selected.reshape(-1).to(dtype=torch.int64),
        )
        self._step_route_total.add_(selected.to(dtype=torch.int64).sum())
        if hot_map is not None:
            # ``safe_ids`` is clamped before this lookup.  ``selected`` still
            # carries the original in-range predicate, so malformed IDs can
            # never contribute while duplicate routing lanes are retained.
            self._step_route_hot.add_(
                (selected & hot_map[safe_ids].ge(0)).to(dtype=torch.int64).sum()
            )
        self._expected_layer.add_(1)

    def finish_step(self, num_tokens: int, *, heat_enabled: bool | None = None) -> None:
        """Finish one complete model step using the known true-token count."""
        num_tokens = _integer("num_tokens", num_tokens, 0)
        if heat_enabled is not None:
            self._set_gate(heat_enabled)

        self._error_flag.logical_or_(self._expected_layer.ne(self.num_layers))
        self._error_flag.logical_or_(self._step_valid_tokens.ne(num_tokens))

        # Counts are cast once into persistent float64 scratch storage.  The
        # heat update remains a separate multiply and add, never a fused or
        # repeated floating scatter operation.
        # Select the exact persistent decay scalar.  Forming ``1 + (decay -
        # 1)`` can round differently for some float64 decays, and a fused
        # multiply-add would change the policy's reference result.
        self._scratch.copy_(self.counts)
        self._step_positive_float.fill_(float(num_tokens > 0))
        active = self._enabled & self._step_positive_float.ne(0)
        self._effective_float.copy_(torch.where(active, self._one, self._zero))
        self._scale.copy_(torch.where(active, self._decay_value, self._one))
        self._scratch.mul_(self._effective_float)
        self.heat.mul_(self._scale)
        self.heat.add_(self._scratch)

        self._step_tokens.fill_(num_tokens)
        self._step_tokens.mul_(self._enabled_int)
        self._tokens_total.add_(self._step_tokens)
        self._last_step_tokens.copy_(self._step_tokens)
        self._forward_count.add_(self._enabled_int)
        self._route_total.add_(
            self._step_route_total
            * self._enabled_int
            * self._step_tokens.ne(0).to(dtype=torch.int64)
        )
        self._route_hot.add_(
            self._step_route_hot
            * self._enabled_int
            * self._step_tokens.ne(0).to(dtype=torch.int64)
        )

        # Cadence uses device integer metadata as well as the host mirror.  It
        # never reads a scalar back during a model step.
        if self.sync_period > 0:
            previous = self._tokens_total - self._step_tokens
            before = torch.div(previous, self.sync_period, rounding_mode="floor")
            after = torch.div(
                self._tokens_total, self.sync_period, rounding_mode="floor"
            )
            crossed = after > before
            self._resync_due.logical_or_(crossed & self._step_tokens.ne(0))

        if self._host_enabled:
            previous_host = self._host_tokens_total
            self._host_forwards += 1
            self._host_last_step_tokens = num_tokens
            if num_tokens:
                self._host_tokens_total += num_tokens
                if self.sync_period > 0:
                    self._host_due = self._host_due or (
                        self._host_tokens_total // self.sync_period
                        > previous_host // self.sync_period
                    )
            else:
                self._host_last_step_tokens = 0

        # Keep availability on device so a captured record sequence still
        # updates it during replay.  Disabled startup forwards do not poison
        # cumulative hot statistics; the device gate remains dynamic here.
        hot_map_incomplete = self._step_hot_map_missing | (
            ~self._step_hot_map_seen.all()
        )
        self._hot_stats_valid.logical_and_(~(self._enabled & hot_map_incomplete))
        self._hot_stats_initialized.logical_or_(self._enabled)

        self.counts.zero_()
        self._expected_layer.zero_()
        self._step_route_total.zero_()
        self._step_route_hot.zero_()
        self._step_hot_map_seen.zero_()
        self._step_hot_map_missing.zero_()

        # A new observation supersedes an unconsumed boundary snapshot.  The
        # importer still rejects an old sequence if a caller attempts to use it.
        if self._pending_snapshot is not None and (
            self._host_forwards != self._pending_snapshot.forwards
            or self._host_tokens_total != self._pending_snapshot.tokens
            or self._host_last_step_tokens != self._pending_snapshot.last_step_tokens
        ):
            self._pending_snapshot = None

    def finish(self, num_tokens: int, *, heat_enabled: bool | None = None) -> None:
        """Compatibility alias for the core model-boundary operation."""
        self.finish_step(num_tokens, heat_enabled=heat_enabled)

    @property
    def has_unreported_observation(self) -> bool:
        return (
            self._host_forwards != self._last_snapshot_forwards
            or self._host_tokens_total != self._last_snapshot_tokens
            or self._host_last_step_tokens != self._last_snapshot_step_tokens
        )

    def should_snapshot(self) -> bool:
        """Return whether a due snapshot may be emitted at this boundary."""
        return self._host_due and self._host_last_step_tokens == 1

    def _copy_heat_to_tuple(self) -> tuple[tuple[float, ...], ...]:
        # This is the only heat D2H path.  The clone and nested tuples ensure
        # no tensor or NumPy storage remains aliased to the live accumulator.
        cpu_heat = self.heat.detach().to(device="cpu").clone()
        return tuple(tuple(float(value) for value in row) for row in cpu_heat.tolist())

    def snapshot(self, *, force: bool = False) -> DeviceSnapshot | None:
        """Copy a due or explicitly requested immutable host snapshot."""
        if not self._host_enabled or not self.has_unreported_observation:
            return self._pending_snapshot
        if not force and not self.should_snapshot():
            return None
        if self._pending_snapshot is not None:
            return self._pending_snapshot

        self._sequence += 1
        invalid = self.error
        route_total_total = int(self._route_total.detach().cpu().item())
        route_hot_total = int(self._route_hot.detach().cpu().item())
        snapshot = DeviceSnapshot(
            heat=self._copy_heat_to_tuple(),
            tokens=self._host_tokens_total,
            forwards=self._host_forwards,
            route_hot=route_hot_total,
            route_total=route_total_total,
            verified=not invalid,
            last_step_tokens=self._host_last_step_tokens,
            base_tokens_total=self._base_tokens_total,
            base_version=self._base_version,
            base_last_sync_tokens=self._base_last_sync_tokens,
            session_id=self.session_id,
            sequence=self._sequence,
            resync_due=self._host_due,
            error=invalid,
            tokens_total=self._host_tokens_total,
            forwards_total=self._host_forwards,
            route_hot_available=self.route_hot_available,
            route_total_total=route_total_total,
        )
        self._pending_snapshot = snapshot
        self._last_snapshot_forwards = self._host_forwards
        self._last_snapshot_tokens = self._host_tokens_total
        self._last_snapshot_route_total = route_total_total
        self._last_snapshot_route_hot = route_hot_total
        self._last_snapshot_step_tokens = self._host_last_step_tokens
        return snapshot

    def flush(self) -> DeviceSnapshot | None:
        """Force one host snapshot at an explicit flush boundary."""
        return self.snapshot(force=True)

    def acknowledge_snapshot(self, snapshot: DeviceSnapshot) -> None:
        """Acknowledge delivery without changing policy-owned cadence state."""
        if snapshot.session_id != self.session_id:
            raise ValueError("snapshot belongs to a different accumulator session")
        if snapshot.sequence is None or snapshot.sequence != self._sequence:
            raise ValueError("snapshot sequence is stale or unknown")
        if self._pending_snapshot is not None and snapshot != self._pending_snapshot:
            raise ValueError("snapshot is not the pending accumulator snapshot")
        self._pending_snapshot = None

    def rebase(
        self,
        *,
        tokens_total: int,
        version: int,
        last_sync_tokens: int,
    ) -> None:
        """Start the next import window from committed policy metadata.

        The coordinator calls this after importing a snapshot and, when
        applicable, committing its plan.  Device heat remains untouched.
        """
        tokens_total = _integer("tokens_total", tokens_total, 0)
        version = _integer("version", version, 0)
        last_sync_tokens = _integer("last_sync_tokens", last_sync_tokens, 0)
        if tokens_total != self._host_tokens_total:
            raise ValueError("rebase token total must match accumulated device heat")
        if last_sync_tokens > tokens_total:
            raise ValueError("last_sync_tokens cannot exceed tokens_total")
        self._base_tokens_total = tokens_total
        self._base_version = version
        self._base_last_sync_tokens = last_sync_tokens
        self._host_last_sync_tokens = last_sync_tokens
        self._last_sync_tokens.fill_(last_sync_tokens)
        self._host_due = self._host_due and last_sync_tokens < self._host_tokens_total
        self._resync_due.fill_(self._host_due)
        self._pending_snapshot = None


class DeviceObserver:
    """Observer adapter exposing the runtime's negotiated observer methods.

    The observer can be configured with ``num_experts`` at construction or at
    ``allocate``/``finish`` time.  When the runtime supplies it only at finish,
    fixed device routing records are retained until that boundary and reduced
    without a host copy.  ``finish`` returns one of this module's
    ``DeviceSnapshot``/``Deferred`` objects; the runtime may wrap those objects
    in its own result dataclasses without importing this module in reverse.
    """

    def __init__(
        self,
        num_layers: int | None = None,
        num_experts: int | None = None,
        *,
        decay: float = 0.999,
        sync_period: int = 50,
        initial_scores: Sequence[Sequence[float]] | torch.Tensor | None = None,
        session_id: str | int | None = None,
    ) -> None:
        self._num_layers = (
            None if num_layers is None else _integer("num_layers", num_layers, 1)
        )
        self._num_experts = (
            None if num_experts is None else _integer("num_experts", num_experts, 1)
        )
        self._decay = _number("decay", decay)
        self._sync_period = _integer("sync_period", sync_period, 0)
        self._initial_scores = initial_scores
        self._session_id = session_id
        self._accumulator: DeviceHeatAccumulator | None = None
        self._ids_record: torch.Tensor | None = None
        self._activity_record: torch.Tensor | None = None
        self._valid_record: torch.Tensor | None = None
        self._device: torch.device | None = None
        self._capacity = 0
        self._top_k = 0
        self._recorded_layers = 0
        # Keep map references rather than copies.  The runtime swaps map
        # contents in place, and a graph replay must observe the current map.
        self._hot_maps: dict[int, torch.Tensor | None] = {}

    @property
    def accumulator(self) -> DeviceHeatAccumulator | None:
        return self._accumulator

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def records(self) -> torch.Tensor | None:
        return self._ids_record

    @property
    def records_host(self) -> None:
        # Device observers intentionally never own a per-forward host record.
        return None

    def _configure_accumulator(
        self,
        device: torch.device,
        num_experts: int,
        *,
        top_k: int | None = None,
        max_rows: int | None = None,
    ) -> DeviceHeatAccumulator:
        if self._accumulator is None:
            if self._num_layers is None:
                raise ValueError(
                    "num_layers is required before configuring the observer"
                )
            self._num_experts = _integer("num_experts", num_experts, 1)
            self._accumulator = DeviceHeatAccumulator(
                self._num_layers,
                self._num_experts,
                decay=self._decay,
                sync_period=self._sync_period,
                initial_scores=self._initial_scores,
                device=device,
                top_k=top_k,
                max_rows=max_rows,
                enabled=False,
                session_id=self._session_id,
            )
        elif self._accumulator.device != device:
            raise ValueError("observer device changed after allocation")
        return self._accumulator

    def allocate(
        self,
        device: torch.device | str,
        layers: int,
        top_k: int,
        max_tokens: int,
        num_experts: int | None = None,
    ) -> None:
        """Allocate fixed device record and heat storage."""
        if self._ids_record is not None:
            raise RuntimeError("device observer records are already allocated")
        layers = _integer("layers", layers, 1)
        top_k = _integer("top_k", top_k, 1)
        max_tokens = _integer("max_tokens", max_tokens, 1)
        if self._num_layers is not None and layers != self._num_layers:
            raise ValueError(
                "allocated layer count differs from observer configuration"
            )
        self._num_layers = layers
        self._top_k = top_k
        self._capacity = max_tokens
        self._device = torch.device(device)
        self._ids_record = torch.zeros(
            (max_tokens, layers, top_k), dtype=torch.int64, device=self._device
        )
        self._activity_record = torch.zeros(
            (max_tokens, layers, top_k), dtype=torch.bool, device=self._device
        )
        self._valid_record = torch.zeros(
            (max_tokens, layers), dtype=torch.bool, device=self._device
        )
        if num_experts is None:
            num_experts = self._num_experts
        if num_experts is not None:
            self._configure_accumulator(
                self._device,
                num_experts,
                top_k=top_k,
                max_rows=max_tokens,
            )

    def record_layer(
        self,
        layer_index: int,
        ids: Any,
        activity_flags: Any = None,
        valid_mask: Any = None,
        rows: Any = None,
        hot_map: Any = None,
    ) -> None:
        """Record one layer in the negotiated or historical argument order."""
        if self._ids_record is None:
            raise RuntimeError("device observer records are not allocated")
        layer_index = _integer("layer_index", layer_index, 0)
        ids, activity_flags, valid_mask, rows, hot_map = (
            DeviceHeatAccumulator._normalise_record_args(
                self, ids, activity_flags, valid_mask, rows, hot_map
            )
        )
        if layer_index >= self._num_layers or rows > self._capacity:
            raise ValueError("record exceeds the observer allocation")
        if ids.ndim != 2 or ids.shape[1] != self._top_k:
            raise ValueError("routing width does not match the observer allocation")
        if activity_flags.shape != ids.shape or valid_mask.ndim != 1:
            raise ValueError("routing record tensors have incompatible shapes")
        if layer_index == 0:
            self._hot_maps.clear()
        self._hot_maps[layer_index] = hot_map
        self._ids_record[:rows, layer_index].copy_(ids[:rows], non_blocking=True)
        self._activity_record[:rows, layer_index].copy_(
            activity_flags[:rows].ne(0), non_blocking=True
        )
        self._valid_record[:rows, layer_index].copy_(
            valid_mask[:rows], non_blocking=True
        )
        self._recorded_layers += 1
        if self._accumulator is not None:
            self._accumulator.record_layer(
                layer_index, ids, activity_flags, valid_mask, rows, hot_map
            )

    def finish(
        self,
        rows: int,
        valid_rows: int | None,
        heat_enabled: bool,
        stream: Any = None,
        num_experts: int | None = None,
    ) -> DeviceSnapshot | Deferred:
        """Finish a model boundary and return a snapshot, legacy result, or defer."""
        rows = _integer("rows", rows, 1)
        if valid_rows is None:
            raise ValueError(
                "device observer requires the runner's known valid token count"
            )
        valid_rows = _integer("valid_rows", valid_rows, 0)
        if valid_rows > rows:
            raise ValueError("valid_rows cannot exceed rows")
        if self._ids_record is None or self._device is None:
            raise RuntimeError("device observer records are not allocated")
        if num_experts is None:
            num_experts = self._num_experts
        if num_experts is None:
            raise ValueError("num_experts is required to finish a device observation")
        was_configured = self._accumulator is not None
        accumulator = self._configure_accumulator(
            self._device,
            num_experts,
            top_k=self._top_k,
            max_rows=self._capacity,
        )
        if not was_configured:
            # If the runtime only recorded fixed buffers during capture, reduce
            # those buffers at the boundary.  Once the accumulator is already
            # configured, direct records own the layer sequence; a partial
            # sequence is left for ``finish_step`` to flag rather than being
            # counted a second time from the mirror buffers.
            for layer in range(self._num_layers):
                accumulator.record_layer(
                    layer,
                    self._ids_record[:rows, layer],
                    self._activity_record[:rows, layer],
                    self._valid_record[:rows, layer],
                    rows,
                    self._hot_maps.get(layer),
                )
        self._recorded_layers = 0
        self._hot_maps.clear()
        accumulator.finish_step(valid_rows, heat_enabled=heat_enabled)
        if not heat_enabled:
            return Deferred()
        if accumulator.should_snapshot():
            snapshot = accumulator.snapshot()
            if snapshot is not None:
                return snapshot
        return Deferred()

    def flush(self) -> DeviceSnapshot | None:
        """Force observation delivery at an explicit report/end boundary."""
        if self._accumulator is None:
            return None
        return self._accumulator.flush()

    def on_heat_enabled(self) -> None:
        """Open the accumulator's device gate after startup."""
        if self._accumulator is not None:
            self._accumulator.on_heat_enabled()

    def acknowledge_snapshot(self, snapshot: DeviceSnapshot) -> None:
        if self._accumulator is None:
            raise RuntimeError("device observer is not configured")
        self._accumulator.acknowledge_snapshot(snapshot)

    def rebase(self, *, tokens_total: int, version: int, last_sync_tokens: int) -> None:
        if self._accumulator is None:
            raise RuntimeError("device observer is not configured")
        self._accumulator.rebase(
            tokens_total=tokens_total,
            version=version,
            last_sync_tokens=last_sync_tokens,
        )
