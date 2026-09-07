# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ple_mmap import (
    _SMALL_GATHER_MAX_ROWS,
    MmapPleTable,
)


@pytest.fixture
def mmap_table(tmp_path):
    row_bytes = 4
    shard_size = 128
    rows = np.arange(256, dtype=np.uint32).view(np.uint8).reshape(256, row_bytes)
    shards = {}
    for shard_idx in range(2):
        path = tmp_path / f"shard-{shard_idx}.bin"
        path.write_bytes(
            rows[shard_idx * shard_size : (shard_idx + 1) * shard_size].tobytes()
        )
        shards[shard_idx] = (str(path), 0, shard_size)

    table = MmapPleTable(
        shards,
        shard_size=shard_size,
        row_bytes=row_bytes,
        torch_dtype=torch.uint8,
        workers=2,
        chunk=2048,
        model_path=str(tmp_path),
        serial=_SMALL_GATHER_MAX_ROWS * 2,
    )
    yield table, rows
    table.close()


def test_small_serial_gather_preserves_order_and_duplicates(mmap_table, monkeypatch):
    table, rows = mmap_table

    def unique_is_not_needed(*args, **kwargs):
        pytest.fail("small SERIAL gather should not call np.unique")

    monkeypatch.setattr(np, "unique", unique_is_not_needed)
    ids = np.array([[1, 250], [130, 250]], dtype=np.int64)

    actual = table.gather(ids)

    np.testing.assert_array_equal(actual, rows[ids.reshape(-1)])
    assert actual.shape == (ids.size, rows.shape[1])
    assert actual.flags.writeable
    assert table._errors == 0

    actual[0, 0] ^= 0xFF
    next_gather = table.gather(ids)
    np.testing.assert_array_equal(next_gather, rows[ids.reshape(-1)])
    assert not np.shares_memory(actual, next_gather)


@pytest.mark.parametrize(
    "ids, expected_range",
    [
        (np.array([-1], dtype=np.int64), "[-1, -1]"),
        (np.array([256], dtype=np.int64), "[256, 256]"),
        (np.array([-1, 256], dtype=np.int64), "[-1, 256]"),
    ],
)
def test_small_serial_gather_preserves_out_of_range_error(
    mmap_table, ids, expected_range
):
    table, _rows = mmap_table

    with pytest.raises(
        IndexError,
        match=rf"PLE mmap: row id out of range \{expected_range} for 256 rows",
    ):
        table.gather(ids)

    assert table._errors == 1


def test_small_gather_cap_keeps_deduplicating_path(mmap_table, monkeypatch):
    table, rows = mmap_table
    unique_called = False
    original_unique = np.unique

    def recording_unique(*args, **kwargs):
        nonlocal unique_called
        unique_called = True
        return original_unique(*args, **kwargs)

    monkeypatch.setattr(np, "unique", recording_unique)
    ids = np.arange(_SMALL_GATHER_MAX_ROWS + 1, dtype=np.int64)

    actual = table.gather(ids)

    np.testing.assert_array_equal(actual, rows[ids])
    assert unique_called


def test_small_gather_falls_back_for_noncontiguous_shard_runs(mmap_table, monkeypatch):
    table, rows = mmap_table
    unique_called = False
    original_unique = np.unique

    def recording_unique(*args, **kwargs):
        nonlocal unique_called
        unique_called = True
        return original_unique(*args, **kwargs)

    monkeypatch.setattr(np, "unique", recording_unique)
    ids = np.array([0, 128, 0, 128], dtype=np.int64)

    actual = table.gather(ids)

    np.testing.assert_array_equal(actual, rows[ids])
    assert unique_called


def test_small_gather_serial_disabled_uses_existing_path(mmap_table, monkeypatch):
    table, rows = mmap_table
    table.serial = 0
    unique_called = False
    original_unique = np.unique

    def recording_unique(*args, **kwargs):
        nonlocal unique_called
        unique_called = True
        return original_unique(*args, **kwargs)

    monkeypatch.setattr(np, "unique", recording_unique)
    ids = np.array([3, 7, 130], dtype=np.int64)

    actual = table.gather(ids)

    np.testing.assert_array_equal(actual, rows[ids])
    assert unique_called


def test_small_gather_readahead_uses_existing_path(mmap_table, monkeypatch):
    table, rows = mmap_table
    table.readahead = 1
    unique_called = False
    original_unique = np.unique

    def recording_unique(*args, **kwargs):
        nonlocal unique_called
        unique_called = True
        return original_unique(*args, **kwargs)

    monkeypatch.setattr(np, "unique", recording_unique)
    ids = np.array([3, 7, 130], dtype=np.int64)

    actual = table.gather(ids)

    np.testing.assert_array_equal(actual, rows[ids])
    assert unique_called


def test_small_gather_empty_returns_fresh_writable_output(mmap_table):
    table, rows = mmap_table

    actual = table.gather(np.empty((0, 2), dtype=np.int64))

    assert actual.shape == (0, rows.shape[1])
    assert actual.dtype == np.uint8
    assert actual.flags.writeable


def test_small_gather_missing_shard_preserves_error(mmap_table):
    table, _rows = mmap_table
    saved_mm = table.mm[1]
    table.mm[1] = None
    try:
        with pytest.raises(IndexError, match="PLE mmap: shard 1 missing"):
            table.gather(np.array([130], dtype=np.int64))
    finally:
        table.mm[1] = saved_mm

    assert table._errors == 1
