# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred host staging for the Qwen4Exp PLE row lookup.

The captured side of a PLE lookup is a fixed-shape copy from pinned host
memory.  This module keeps the request-dependent row IDs and gathered rows in
stable pinned buffers, and uses CUDA stream memops to let a captured
consumer wait for the host gather without a device synchronization.

The flag protocol and extension API are reused from FreeToken's Apache-2.0
implementation at
``python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp`` (FreeToken
commit ``af71ba43206e124f5ff6419b47ee36c6e9981078``).  The extension is
loaded lazily so importing vLLM on a CPU-only host does not require CUDA or a
separate FreeToken build.
"""

from __future__ import annotations

import importlib
from contextlib import suppress
from typing import Any

import torch

_EXTENSION_NAME = "vllm._ple_memops"
_EXTENSION_CALLS = (
    "memop_wait_reset",
    "memop_write",
    "memop_wait_geq",
    "signal_flag",
)
_extension: Any | None = None


class StreamMemopsUnavailable(RuntimeError):
    """The optional CUDA stream-memop capability is unavailable at startup."""


def _load_extension() -> Any:
    """Load the optional stream-memop extension with an actionable error."""
    global _extension
    if _extension is not None:
        return _extension
    try:
        module = importlib.import_module(_EXTENSION_NAME)
    except (ImportError, OSError) as exc:
        raise StreamMemopsUnavailable(
            f"Qwen4Exp deferred PLE requires a CUDA vLLM build with {_EXTENSION_NAME}."
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
    """Get the CUDA stream handle accepted by the in-tree CUDA extension."""
    try:
        return int(stream.cuda_stream)
    except AttributeError:
        return int(stream)


def _check_memop_status(status: Any, operation: str) -> None:
    """Report a rejected CUDA driver operation without hiding its status."""
    if status is not None and int(status) != 0:
        raise RuntimeError(
            f"PLE stream operation {operation} failed with CUDA status {status}"
        )


class DeferredRows:
    """Stage one fixed-shape PLE row batch across a captured forward.

    ``destination`` is the stable device staging tensor consumed by the
    captured model.  It must be shaped ``[1, heads, head_dim]`` and remain
    alive for the lifetime of this object.  ``table`` must provide
    ``gather(np.ndarray)`` and return one row per flattened ID.  The table is
    intentionally kept on the host side: no table read or allocation occurs
    during capture.

    The normal sequence is::

        rows.prepare(ids_cuda)
        dispatch_forward()
        rows.complete()
        rows.consume()  # called by the captured forward

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
        if destination.ndim != 3 or destination.shape[0] != 1:
            raise ValueError(
                "DeferredRows destination must have shape [1, heads, head_dim]"
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
        self._stream_override = stream
        self.stream = _stream_for(destination.device, stream)
        shape = tuple(int(size) for size in destination.shape)
        self.ids = torch.empty(
            (1, shape[1]), dtype=torch.int64, device="cpu", pin_memory=True
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

    def _validate_ids(self, ids: torch.Tensor) -> None:
        if not isinstance(ids, torch.Tensor):
            raise TypeError("DeferredRows IDs must be a torch.Tensor")
        if ids.device != self.destination.device:
            raise ValueError(
                f"DeferredRows IDs device {ids.device} != "
                f"destination device {self.destination.device}"
            )
        expected = (1, int(self.destination.shape[1]))
        if tuple(ids.shape) != expected:
            raise ValueError(
                f"DeferredRows IDs shape {tuple(ids.shape)} != expected {expected}"
            )
        if ids.dtype != torch.int64:
            raise ValueError(f"DeferredRows IDs must be torch.int64, got {ids.dtype}")
        if not ids.is_contiguous():
            raise ValueError("DeferredRows IDs must be contiguous")

    def _validate_destination(self, destination: torch.Tensor) -> None:
        if not isinstance(destination, torch.Tensor):
            raise TypeError("DeferredRows destination must be a torch.Tensor")
        if destination is not self.destination:
            if destination.device != self.destination.device:
                raise ValueError(
                    f"DeferredRows destination device {destination.device} != "
                    f"{self.destination.device}"
                )
            if tuple(destination.shape) != tuple(self.destination.shape):
                raise ValueError(
                    f"DeferredRows destination shape {tuple(destination.shape)} "
                    f"!= {tuple(self.destination.shape)}"
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
        """Probe outside capture, retaining scratch until queued writes finish."""
        scratch = torch.zeros((1,), dtype=torch.int64, device="cpu", pin_memory=True)
        stream_ptr = _stream_ptr(stream)
        try:
            for operation, function in (
                ("write", self._ext.memop_write),
                ("wait", self._ext.memop_wait_geq),
            ):
                status = function(stream_ptr, int(scratch.data_ptr()), 7)
                # CUDA_ERROR_NOT_SUPPORTED. Other failures are not a safe
                # reason to silently select a different execution path.
                if status == 801:
                    raise StreamMemopsUnavailable(
                        f"CUDA stream memop {operation} is not supported"
                    )
                _check_memop_status(status, f"{operation}(probe, 7)")
        finally:
            stream.synchronize()
        if int(scratch[0]) != 7:
            raise RuntimeError(
                "FreeToken PLE stream memop probe did not publish its value"
            )

    def prepare(self, ids: torch.Tensor) -> None:
        """Queue the fixed-shape ID readback before dispatching the forward.

        The stream write of zero is deliberately queued before the D2H copy.
        A host ``flag.zero_`` would race an earlier graph replay and can leave
        the next replay waiting on a value that belongs to the wrong batch.
        """
        self._ensure_healthy()
        if self._pending:
            raise RuntimeError("DeferredRows.prepare called while a batch is pending")
        self._validate_ids(ids)
        stream = self._operation_stream()
        try:
            status = self._ext.memop_write(_stream_ptr(stream), self.flag_ptr, 0)
            _check_memop_status(status, "memop_write(flag, 0)")
            self._prepare_stream = stream
            self._reset_queued = True
            self.ids.copy_(ids, non_blocking=True)
            self._readback_event.record(stream)
            self._readback_recorded = True
            self._pending = True
            self._rows_ready = False
            self._gate_armed = False
        except BaseException as exc:
            self._poison(exc)
            raise

    def _as_rows_tensor(self, gathered: Any) -> torch.Tensor:
        """Convert a table result to the destination's dtype and fixed shape."""
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
        target_nbytes = self.rows.numel() * self.rows.element_size()
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
            source = source.reshape(-1).view(self.rows.dtype)
        elif source.numel() == self.rows.numel():
            source = source.to(dtype=self.rows.dtype)
        else:
            raise ValueError(
                "DeferredRows table returned a non-byte result with an "
                "incompatible element count"
            )
        return source.reshape(self.rows.shape)

    def complete(self) -> None:
        """Finish the host gather and release the captured consumer."""
        self._ensure_healthy()
        if not self._pending:
            raise RuntimeError("DeferredRows.complete called without prepare")
        try:
            self._readback_event.synchronize()
            ids = self.ids.numpy().reshape(-1)
            gathered = self.table.gather(ids)
            self.rows.copy_(self._as_rows_tensor(gathered))
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
                status = self._ext.memop_wait_reset(_stream_ptr(stream), self.flag_ptr)
                _check_memop_status(status, "memop_wait_reset")
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
        """Copy the completed or dummy rows into the fixed device staging.

        Under CUDA graph capture, the wait/reset is emitted before the H2D
        copy.  ``wait=False`` is available when a caller already invoked
        :meth:`gate` for a split implementation.
        """
        self._ensure_healthy()
        if self._pending:
            raise RuntimeError("DeferredRows.consume called while a batch is pending")
        if not self._rows_ready:
            raise RuntimeError("DeferredRows.consume has no completed rows")
        destination = self.destination if destination is None else destination
        self._validate_destination(destination)
        stream = self._operation_stream(stream)
        if capture is None:
            capture = bool(torch.cuda.is_current_stream_capturing())
        try:
            if wait:
                if self._gate_armed:
                    self._gate_armed = False
                elif capture:
                    status = self._ext.memop_wait_reset(
                        _stream_ptr(stream), self.flag_ptr
                    )
                    _check_memop_status(status, "memop_wait_reset")
                else:
                    # Keep the eager path harmless after a caller reused a
                    # helper without a preceding captured wait.
                    self._signal()
            destination.copy_(self.rows, non_blocking=True)
            return destination
        except BaseException as exc:
            self._poison(exc)
            raise

    def prepare_dummy(self) -> None:
        """Reset pinned rows for warmup/capture after the prior copy finishes."""
        self._ensure_healthy()
        if self._pending:
            raise RuntimeError("DeferredRows.prepare_dummy called while pending")
        try:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("DeferredRows.prepare_dummy cannot run in capture")
            self._operation_stream().synchronize()
            self.rows.zero_()
            self._signal()
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
    "PLEDeferredStaging",
    "PLEWait",
]
