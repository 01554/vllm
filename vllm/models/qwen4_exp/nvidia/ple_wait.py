# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred host staging for the Qwen4Exp PLE row lookup.

The captured side of a PLE lookup is a fixed-shape copy from pinned host
memory.  This module keeps the request-dependent row IDs and gathered rows in
stable pinned buffers, and uses FreeToken's stream memops to let a captured
consumer wait for the host gather without a device synchronization.

The flag protocol and extension API are reused from FreeToken's Apache-2.0
implementation at
``python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp`` (FreeToken
commit ``af71ba43206e124f5ff6419b47ee36c6e9981078``).  The extension is
loaded lazily so importing vLLM on a CPU-only host does not require CUDA or a
FreeToken build.
"""

from __future__ import annotations

import importlib
from contextlib import suppress
from typing import Any

import torch

_EXTENSION_NAME = "vllm_ple_wait_ext"
_EXTENSION_CALLS = (
    "memop_wait_reset",
    "memop_write",
    "memop_wait_geq",
    "signal_flag",
)
# Keep the small deferred capacity separate from the model's potentially much
# larger mmap staging allocation so deferred pinned memory stays bounded.
PLE_DEFERRED_MAX_TOKENS = 8
_extension: Any | None = None


def _load_extension() -> Any:
    """Load the optional stream-memop extension with an actionable error."""
    global _extension
    if _extension is not None:
        return _extension
    try:
        module = importlib.import_module(_EXTENSION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Qwen4Exp deferred PLE requires the standalone "
            f"{_EXTENSION_NAME} extension. Build it with "
            "tests/models/qwen4_exp/build_ple_wait_extension.py and expose "
            "its output directory on PYTHONPATH."
        ) from exc
    missing = [
        name for name in _EXTENSION_CALLS if not callable(getattr(module, name, None))
    ]
    if missing:
        raise RuntimeError(
            f"{_EXTENSION_NAME} is missing required callables: {', '.join(missing)}"
        )
    _extension = module
    return module


def _stream_for(device: torch.device, stream: Any | None) -> Any:
    """Return a CUDA stream, accepting the torch stream wrapper used by vLLM."""
    if stream is not None:
        return stream
    return torch.cuda.current_stream(device)


def _stream_ptr(stream: Any) -> int:
    """Get the CUDA stream handle accepted by the FreeToken extension."""
    try:
        return int(stream.cuda_stream)
    except AttributeError:
        return int(stream)


def _check_memop_status(status: Any, operation: str) -> None:
    """Turn the extension's ``-1`` unavailable status into a clear error."""
    if status is not None and int(status) != 0:
        raise RuntimeError(
            f"FreeToken PLE stream operation {operation} was rejected; "
            "CUDA stream memops are unavailable"
        )


class DeferredRows:
    """Stage a small fixed-capacity PLE row batch across a captured forward.

    ``destination`` is the stable device staging tensor consumed by the
    captured model.  It must be shaped ``[rows, heads, head_dim]`` and remain
    alive for the lifetime of this object. ``rows`` is normally at most
    :data:`PLE_DEFERRED_MAX_TOKENS`. ``table`` must provide
    ``gather(np.ndarray)`` and return one row per flattened ID.  The table is
    intentionally kept on the host side: no table read or allocation occurs
    during capture.

    The normal sequence is::

        rows.prepare(ids_cuda, padded_rows=graph_rows)
        dispatch_forward()
        rows.complete()
        rows.consume(output)  # called by the captured forward

    ``prepare_dummy`` supplies zero rows and a pre-signaled flag for warmup or
    graph capture.  ``abort`` unblocks a pending consumer before poisoning the
    object, so an exception cannot leave a CUDA graph waiting forever.
    """

    def __init__(
        self,
        destination: torch.Tensor,
        table: Any,
        *,
        stream: Any | None = None,
    ) -> None:
        if not isinstance(destination, torch.Tensor):
            raise TypeError("DeferredRows destination must be a torch.Tensor")
        if destination.device.type != "cuda":
            raise ValueError("DeferredRows destination must be a CUDA tensor")
        if destination.ndim != 3 or destination.shape[0] <= 0:
            raise ValueError(
                "DeferredRows destination must have shape "
                "[rows, heads, head_dim] with rows > 0"
            )
        if any(int(size) <= 0 for size in destination.shape[1:]):
            raise ValueError("DeferredRows destination dimensions must be positive")
        if not destination.is_contiguous():
            raise ValueError("DeferredRows destination must be contiguous")
        if not callable(getattr(table, "gather", None)):
            raise TypeError("DeferredRows table must provide gather(ids_numpy)")

        # Resolve this before allocating the pinned buffers. An absent opt-in
        # extension should fail at construction, rather than much later from
        # inside a graph replay.
        self._ext = _load_extension()
        self.destination = destination
        self.table = table
        self.capacity = int(destination.shape[0])
        self._stream_override = stream
        self.stream = _stream_for(destination.device, stream)
        shape = tuple(int(size) for size in destination.shape)
        self.ids = torch.empty(
            (self.capacity, shape[1]), dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.rows = torch.empty(
            shape, dtype=destination.dtype, device="cpu", pin_memory=True
        )
        self.flag = torch.empty((1,), dtype=torch.int64, device="cpu", pin_memory=True)
        self.ids.zero_()
        self.rows.zero_()
        # A freshly allocated helper represents a dummy-ready staging buffer.
        # This is required for a warmup/capture path that consumes before a
        # real host gather has completed.
        self.flag.fill_(1)
        self._probe_stream_memops(self.stream)
        self._readback_event = torch.cuda.Event()
        self._pending = False
        self._rows_ready = True
        self._active_rows = self.capacity
        self._padded_rows = self.capacity
        self._gate_armed = False
        self._reset_queued = False
        self._readback_recorded = False
        self._prepare_stream: Any | None = None
        self._poisoned = False
        self._poison_reason: BaseException | None = None

    @classmethod
    def allocate(
        cls,
        destination: torch.Tensor,
        table: Any,
        *,
        stream: Any | None = None,
    ) -> DeferredRows:
        """Construct a helper, retaining the stable destination address."""
        return cls(destination, table, stream=stream)

    @property
    def pending(self) -> bool:
        """Whether a prepared ID batch still needs :meth:`complete`."""
        return self._pending

    @property
    def poisoned(self) -> bool:
        """Whether a failed operation permanently disabled this helper."""
        return self._poisoned

    @property
    def poison_reason(self) -> BaseException | None:
        """Return the first exception that poisoned this helper, if any."""
        return self._poison_reason

    @property
    def flag_ptr(self) -> int:
        """Pinned flag address for diagnostics and integration tests."""
        return int(self.flag.data_ptr())

    @property
    def ids_pinned(self) -> torch.Tensor:
        """Pinned host ID readback buffer."""
        return self.ids

    @property
    def rows_pinned(self) -> torch.Tensor:
        """Pinned host row buffer."""
        return self.rows

    def _ensure_healthy(self) -> None:
        if self._poisoned:
            detail = f": {self._poison_reason}" if self._poison_reason else ""
            raise RuntimeError(f"DeferredRows is permanently poisoned{detail}")

    def _validate_ids(self, ids: torch.Tensor) -> int:
        if not isinstance(ids, torch.Tensor):
            raise TypeError("DeferredRows IDs must be a torch.Tensor")
        if ids.device != self.destination.device:
            raise ValueError(
                f"DeferredRows IDs device {ids.device} != "
                f"destination device {self.destination.device}"
            )
        if ids.ndim != 2 or ids.shape[1] != self.destination.shape[1]:
            raise ValueError(
                "DeferredRows IDs must have shape "
                f"[1..{self.capacity}, {int(self.destination.shape[1])}], "
                f"got {tuple(ids.shape)}"
            )
        actual_rows = int(ids.shape[0])
        if not 0 < actual_rows <= self.capacity:
            raise ValueError(
                f"DeferredRows IDs row count {actual_rows} outside [1, {self.capacity}]"
            )
        if ids.dtype != torch.int64:
            raise ValueError(f"DeferredRows IDs must be torch.int64, got {ids.dtype}")
        if not ids.is_contiguous():
            raise ValueError("DeferredRows IDs must be contiguous")
        return actual_rows

    def _validate_destination(self, destination: torch.Tensor) -> None:
        if not isinstance(destination, torch.Tensor):
            raise TypeError("DeferredRows destination must be a torch.Tensor")
        if destination.device != self.destination.device:
            raise ValueError(
                f"DeferredRows destination device {destination.device} != "
                f"{self.destination.device}"
            )
        if destination.ndim != 3 or destination.shape[0] <= 0:
            raise ValueError(
                "DeferredRows destination must have shape "
                "[rows, heads, head_dim] with rows > 0"
            )
        if destination.shape[0] > self.capacity:
            raise ValueError(
                f"DeferredRows destination has {int(destination.shape[0])} rows, "
                f"capacity is {self.capacity}"
            )
        if tuple(destination.shape[1:]) != tuple(self.destination.shape[1:]):
            raise ValueError(
                f"DeferredRows destination shape {tuple(destination.shape)} "
                f"has incompatible row shape {tuple(self.destination.shape)}"
            )
        if destination.shape[0] < self._padded_rows:
            raise ValueError(
                f"DeferredRows destination has {int(destination.shape[0])} rows, "
                f"but the prepared graph requires {self._padded_rows}"
            )
        if destination.dtype != self.destination.dtype:
            raise ValueError(
                f"DeferredRows destination dtype {destination.dtype} != "
                f"{self.destination.dtype}"
            )
        if not destination.is_contiguous():
            raise ValueError("DeferredRows destination must be contiguous")

    def _signal(self) -> None:
        """Signal the host flag, accepting only the extension's API."""
        self._ext.signal_flag(self.flag_ptr)

    def _operation_stream(self, stream: Any | None = None) -> Any:
        """Resolve an operation stream, honoring an optional constructor pin."""
        selected = self._stream_override if stream is None else stream
        resolved = _stream_for(self.destination.device, selected)
        self.stream = resolved
        return resolved

    def _probe_stream_memops(self, stream: Any) -> None:
        """Fail closed when the standalone extension cannot gate a stream."""
        scratch = torch.zeros((1,), dtype=torch.int64, device="cpu", pin_memory=True)
        stream_ptr = _stream_ptr(stream)
        _check_memop_status(
            self._ext.memop_write(stream_ptr, int(scratch.data_ptr()), 7),
            "memop_write(probe, 7)",
        )
        _check_memop_status(
            self._ext.memop_wait_geq(stream_ptr, int(scratch.data_ptr()), 7),
            "memop_wait_geq(probe, 7)",
        )
        stream.synchronize()
        if int(scratch[0]) != 7:
            raise RuntimeError(
                "FreeToken PLE stream memop probe did not publish its value"
            )

    def prepare(self, ids: torch.Tensor, padded_rows: int | None = None) -> None:
        """Queue actual ID readback before dispatching the padded forward.

        The stream write of zero is deliberately queued before the D2H copy.
        A host ``flag.zero_`` would race an earlier graph replay and can leave
        the next replay waiting on a value that belongs to the wrong batch.
        ``ids`` contains only real candidate rows. ``padded_rows`` describes
        the graph output width and is used for validation; padding IDs are
        never copied or passed to the mmap table.
        """
        self._ensure_healthy()
        if self._pending:
            raise RuntimeError("DeferredRows.prepare called while a batch is pending")
        actual_rows = self._validate_ids(ids)
        if padded_rows is None:
            padded_rows = actual_rows
        padded_rows = int(padded_rows)
        if not 0 < padded_rows <= self.capacity:
            raise ValueError(
                f"DeferredRows padded row count {padded_rows} outside "
                f"[1, {self.capacity}]"
            )
        if actual_rows > padded_rows:
            raise ValueError(
                f"DeferredRows actual row count {actual_rows} exceeds "
                f"padded row count {padded_rows}"
            )
        stream = self._operation_stream()
        try:
            status = self._ext.memop_write(_stream_ptr(stream), self.flag_ptr, 0)
            _check_memop_status(status, "memop_write(flag, 0)")
            self._prepare_stream = stream
            self._reset_queued = True
            self._active_rows = actual_rows
            self._padded_rows = padded_rows
            self.ids[:actual_rows].copy_(ids, non_blocking=True)
            self.ids[actual_rows:].zero_()
            self._readback_event.record(stream)
            self._readback_recorded = True
            self._pending = True
            self._rows_ready = False
            self._gate_armed = False
        except BaseException as exc:
            self._poison(exc)
            raise

    def _as_rows_tensor(
        self, gathered: Any, num_rows: int | None = None
    ) -> torch.Tensor:
        """Convert a table result to the destination's dtype and row shape."""
        if num_rows is None:
            num_rows = self.capacity
        num_rows = int(num_rows)
        if not 0 < num_rows <= self.capacity:
            raise ValueError(
                f"DeferredRows row count {num_rows} outside [1, {self.capacity}]"
            )
        target = self.rows[:num_rows]
        if isinstance(gathered, torch.Tensor):
            source = gathered.detach()
            if source.device.type != "cpu":
                raise ValueError(
                    "DeferredRows table must return host rows; CUDA table "
                    "results are not copied during completion"
                )
        else:
            # MmapPleTable.gather returns a writable NumPy uint8 array.  The
            # conversion intentionally stays on the host and avoids a second
            # device allocation during the deferred completion callback.
            source = torch.as_tensor(gathered, device="cpu")
        source = source.contiguous()
        target_nbytes = target.numel() * target.element_size()
        source_nbytes = source.numel() * source.element_size()
        if source_nbytes != target_nbytes:
            raise ValueError(
                f"DeferredRows table returned {source_nbytes} bytes, expected "
                f"{target_nbytes}"
            )
        # Mmap tables are raw bytes. Reinterpret them for every destination
        # dtype, including BF16 where one row occupies twice as many bytes as
        # its element count. A tensor table result is converted only when it
        # already has the destination element count.
        if source.dtype == torch.uint8:
            source = source.reshape(-1).view(target.dtype)
        elif source.numel() == target.numel():
            source = source.to(dtype=target.dtype)
        else:
            raise ValueError(
                "DeferredRows table returned a non-byte result with an "
                "incompatible element count"
            )
        return source.reshape(target.shape)

    def complete(self) -> None:
        """Finish the host gather and release the captured consumer."""
        self._ensure_healthy()
        if not self._pending:
            raise RuntimeError("DeferredRows.complete called without prepare")
        try:
            self._readback_event.synchronize()
            ids = self.ids[: self._active_rows].numpy().reshape(-1)
            gathered = self.table.gather(ids)
            self.rows[: self._active_rows].copy_(
                self._as_rows_tensor(gathered, self._active_rows)
            )
            # Clear every row outside the real candidate prefix. This avoids
            # exposing a larger batch's rows when the next graph is smaller.
            self.rows[self._active_rows :].zero_()
            self._signal()
            self._reset_queued = False
            self._readback_recorded = False
            self._prepare_stream = None
            self._pending = False
            self._rows_ready = True
            self._gate_armed = False
        except BaseException as exc:
            self._poison(exc)
            raise

    def gate(
        self,
        *,
        stream: Any | None = None,
        capture: bool | None = None,
        dummy: bool = False,
    ) -> None:
        """Apply the flag operation for integrations that split wait/copy.

        A real captured path emits ``WAIT(>=1); RESET``.  Eager and dummy
        paths use a host signal because they do not need a device wait.  The
        regular :meth:`consume` method invokes this operation itself.
        """
        self._ensure_healthy()
        if self._pending:
            raise RuntimeError("DeferredRows.gate called while a batch is pending")
        if dummy:
            self.prepare_dummy()
            return
        stream = self._operation_stream(stream)
        if capture is None:
            capture = bool(torch.cuda.is_current_stream_capturing())
        try:
            if capture:
                self._ext.memop_wait_reset(_stream_ptr(stream), self.flag_ptr)
                self._gate_armed = True
            else:
                self._signal()
                self._gate_armed = False
        except BaseException as exc:
            self._poison(exc)
            raise

    def consume(
        self,
        destination: torch.Tensor | None = None,
        *,
        stream: Any | None = None,
        capture: bool | None = None,
        wait: bool = True,
    ) -> torch.Tensor:
        """Copy completed or dummy rows into the graph's device staging.

        Under CUDA graph capture, the wait/reset is emitted before the H2D
        copy.  ``wait=False`` is available when a caller already invoked
        :meth:`gate` for a split implementation.
        """
        self._ensure_healthy()
        if self._pending:
            raise RuntimeError("DeferredRows.consume called while a batch is pending")
        if not self._rows_ready:
            raise RuntimeError("DeferredRows.consume has no completed rows")
        if destination is None:
            destination = self.destination[: self._padded_rows]
        self._validate_destination(destination)
        stream = self._operation_stream(stream)
        if capture is None:
            capture = bool(torch.cuda.is_current_stream_capturing())
        try:
            if wait:
                if self._gate_armed:
                    self._gate_armed = False
                elif capture:
                    self._ext.memop_wait_reset(_stream_ptr(stream), self.flag_ptr)
                else:
                    # Keep the eager path harmless after a caller reused a
                    # helper without a preceding captured wait.
                    self._signal()
            destination.copy_(self.rows[: destination.shape[0]], non_blocking=True)
            return destination
        except BaseException as exc:
            self._poison(exc)
            raise

    def prepare_dummy(self, padded_rows: int | None = None) -> None:
        """Reset pinned rows for warmup/capture after the prior copy finishes."""
        self._ensure_healthy()
        if self._pending:
            raise RuntimeError("DeferredRows.prepare_dummy called while pending")
        if padded_rows is None:
            padded_rows = self.capacity
        padded_rows = int(padded_rows)
        if not 0 < padded_rows <= self.capacity:
            raise ValueError(
                f"DeferredRows padded row count {padded_rows} outside "
                f"[1, {self.capacity}]"
            )
        try:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("DeferredRows.prepare_dummy cannot run in capture")
            self._operation_stream().synchronize()
            self.rows.zero_()
            self._signal()
            self._active_rows = 0
            self._padded_rows = padded_rows
            self._rows_ready = True
            self._gate_armed = False
        except BaseException as exc:
            self._poison(exc)
            raise

    def _poison(self, exc: BaseException) -> None:
        """Unblock a possible wait, then permanently fail closed."""
        if self._poisoned:
            return
        # ``prepare`` queues the flag reset before recording the readback
        # event. Do not signal the host flag ahead of that reset: a later
        # queued reset could overwrite the release and strand a graph wait.
        # The readback event is recorded before graph launch, so synchronizing
        # it cannot wait on the captured consumer. If recording failed,
        # prepare is still outside capture and synchronizing its stream is the
        # only safe recovery fence.
        if self._reset_queued:
            try:
                if self._readback_recorded:
                    self._readback_event.synchronize()
                elif self._prepare_stream is not None:
                    self._prepare_stream.synchronize()
            except BaseException:
                # Preserve the original failure and still attempt the host
                # release below. The object is poisoned either way.
                pass
        try:
            self._signal()
        except BaseException:
            # The original exception is more useful to the caller. A best
            # effort host store keeps a graph from waiting if the extension's
            # signal wrapper itself failed after a partial operation.
            with suppress(BaseException):
                self.flag.fill_(1)
        self._reset_queued = False
        self._readback_recorded = False
        self._prepare_stream = None
        self._pending = False
        self._rows_ready = False
        self._gate_armed = False
        self._poisoned = True
        self._poison_reason = exc

    def abort(self) -> None:
        """Signal any pending consumer and permanently poison this helper."""
        if self._poisoned:
            return
        self._poison(RuntimeError("DeferredRows aborted"))


# Keep the earlier coordination name available while the integration uses the
# more descriptive DeferredRows constructor.
PLEWait = DeferredRows
PLEDeferredStaging = DeferredRows


__all__ = [
    "DeferredRows",
    "PLE_DEFERRED_MAX_TOKENS",
    "PLEDeferredStaging",
    "PLEWait",
]
