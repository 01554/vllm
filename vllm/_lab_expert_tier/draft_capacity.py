# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-model VRAM accounting for the expert tier.

The tier budget (`GIB`) is the expert budget only. A draft model (MTP)
loads after the target and after the tier has allocated, so its resident
bytes come on top of that budget inside the same device. This module
gives the two numbers the operator needs, without an env value standing
in for either:

- `estimate_draft_bytes`: before the draft loads, from the checkpoint's
  safetensors headers (tensors under the draft prefix), reported as the
  reservation;
- `measure_resident_bytes`: after the draft loads, the unique device
  storage of its parameters and buffers, with storage shared with the
  target (embedding, lm_head) counted separately and not charged twice.

`VRAM_BUDGET_GIB` (optional) is a check, not a source of bytes: when set,
tier bytes + the estimate must fit it before the draft loads.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Iterable
from typing import Any

INDEX_FILE = "model.safetensors.index.json"


def estimate_draft_bytes(checkpoint: str, prefix: str) -> dict[str, Any]:
    """Sum checkpoint bytes of tensors whose name starts with `prefix`.

    Reads only the safetensors headers of the shards the index maps a
    matching tensor to. Returns totals by dtype, the tensor count and the
    shards read; raises when the index or a shard header is unreadable
    or when no tensor matches.
    """
    index_path = os.path.join(checkpoint, INDEX_FILE)
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    shards = sorted({f for name, f in weight_map.items() if name.startswith(prefix)})
    if not shards:
        raise ValueError(f"No checkpoint tensor starts with {prefix!r}")
    total, count = 0, 0
    by_dtype: dict[str, int] = {}
    for shard in shards:
        for name, entry in _read_header(os.path.join(checkpoint, shard)).items():
            if name == "__metadata__" or not name.startswith(prefix):
                continue
            begin, end = entry["data_offsets"]
            size = int(end) - int(begin)
            total += size
            count += 1
            by_dtype[entry["dtype"]] = by_dtype.get(entry["dtype"], 0) + size
    return {
        "prefix": prefix,
        "bytes": total,
        "tensors": count,
        "by_dtype": dict(sorted(by_dtype.items())),
        "shards": shards,
    }


def _read_header(path: str) -> dict[str, Any]:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def measure_resident_bytes(model: Any, shared_with: Any = None) -> dict[str, Any]:
    """Unique device storage of `model`, split from storage shared with
    `shared_with` (another module) so shared tensors are charged once."""
    foreign: set[int] = set()
    if shared_with is not None:
        for tensor in _tensors(shared_with):
            foreign.add(tensor.untyped_storage().data_ptr())
    seen: dict[int, int] = {}
    shared: dict[int, int] = {}
    by_dtype: dict[str, int] = {}
    host = 0
    for tensor in _tensors(model):
        storage = tensor.untyped_storage()
        if tensor.device.type == "cpu":
            host += storage.nbytes()
            continue
        key = storage.data_ptr()
        if key in foreign:
            shared[key] = storage.nbytes()
        elif key not in seen:
            seen[key] = storage.nbytes()
            dtype = str(tensor.dtype).removeprefix("torch.")
            by_dtype[dtype] = by_dtype.get(dtype, 0) + storage.nbytes()
    return {
        "unique_bytes": sum(seen.values()),
        "shared_bytes": sum(shared.values()),
        "host_bytes": host,
        "storages": len(seen),
        "by_dtype": dict(sorted(by_dtype.items())),
    }


def _tensors(module: Any) -> Iterable[Any]:
    yield from module.parameters()
    yield from module.buffers()


def check_estimate(estimate: dict[str, Any], measured: dict[str, Any], tolerance):
    """Measured unique bytes must not exceed the reservation by more than
    `tolerance` (fraction); a shortfall is reported, never an error."""
    reserved = int(estimate["bytes"])
    unique = int(measured["unique_bytes"])
    excess = unique - reserved
    if excess > tolerance * reserved:
        raise RuntimeError(
            f"Draft resident bytes {unique} exceed the reservation {reserved}"
        )
    return {"reserved_bytes": reserved, "excess_bytes": excess}


__all__ = [
    "INDEX_FILE",
    "check_estimate",
    "estimate_draft_bytes",
    "measure_resident_bytes",
]
