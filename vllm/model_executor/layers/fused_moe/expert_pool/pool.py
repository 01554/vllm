# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shared VRAM bank, its staging rows, the pool tables and layer offsets."""

from __future__ import annotations

from vllm.model_executor.layers.fused_moe.expert_pool.copy import copy_rows
from vllm.model_executor.layers.fused_moe.expert_pool.tables import (
    TENSORS,
    allocate_global_tables,
    check_global_tables,
    read_control,
    resident_per_layer,
    set_control,
)


class GlobalPool:
    """The shared bank, its staging views, the tables, and the layer offsets."""

    def __init__(self, device, sources, slots_per_layer, staging):
        # `sources`: one layer's six tensors in the bank's final layout; only
        # shapes/dtypes are read here (rows are filled by the layers).
        import torch

        self.slots_per_layer = list(slots_per_layer)
        self.staging_slots = staging
        self.tables = allocate_global_tables(
            device, sources[TENSORS[0]].shape[0], self.slots_per_layer, staging
        )
        self.rows = self.tables.pool_rows + staging
        self.offsets = [0]
        for slots in self.slots_per_layer[:-1]:
            self.offsets.append(self.offsets[-1] + slots)
        self.bank = {
            name: torch.zeros(
                (self.rows, *source.shape[1:]), dtype=source.dtype, device=device
            )
            for name, source in sources.items()
        }
        self.staging = {
            name: tensor[self.tables.pool_rows :] for name, tensor in self.bank.items()
        }
        self.row_bytes = sum(t[0].numel() * t.element_size() for t in sources.values())
        self.staging_bytes = self.row_bytes * staging
        self.pool_bytes = self.row_bytes * self.tables.pool_rows

    def offset(self, layer):
        return self.offsets[layer]

    def host_swap(self, layer, old_expert, new_expert):
        """Gate-closed exchange for init verification: `new_expert` takes the
        row of resident `old_expert`, which falls back to its RAM row. The
        caller copies the bytes."""
        tables = self.tables
        if int(tables.gate[0]):
            raise RuntimeError("Host swaps are only allowed while the gate is closed")
        E = tables.num_experts
        old_key, new_key = layer * E + old_expert, layer * E + new_expert
        row = int(tables.hot_phys[old_key])
        if row < 0 or int(tables.hot_phys[new_key]) >= 0:
            raise AssertionError("Swap does not match the current pool placement")
        tables.hot_phys[old_key], tables.cold_phys[old_key] = -1, old_expert
        tables.hot_phys[new_key], tables.cold_phys[new_key] = row, -1
        tables.row_key[row] = new_key

    def snapshot(self):
        """Validate the pool on the host; one copy per stats report."""
        check_global_tables(self.tables)
        return resident_per_layer(self.tables)

    def apply_control(self, **values):
        """Validate then write the controls; the gate is not touched here."""
        return set_control(self.tables, **values)

    def control(self):
        return read_control(self.tables)


def copy_in(source, bank, buffers):
    """Copy the planned host rows of this layer into the bank rows."""
    copy_rows(
        source, bank, buffers.gather_src, buffers.gather_dst, buffers.gather_count
    )
