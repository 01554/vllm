# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""device_loading_context restores CPU-resident parameters by name after
processing; only a replacement explicitly marked DEVICE_RESIDENT_ATTR (the
expert cache's scale slot buffers) stays on the device. Ordinary same-name
replacements and unreplaced parameters are moved back to the CPU as before.
The UVA-offload branch is unchanged and not covered here (cache plus UVA
offload is not a supported combination by this fix)."""

import pytest
import torch

from vllm.model_executor.model_loader.utils import (
    DEVICE_RESIDENT_ATTR,
    device_loading_context,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")


def test_only_marked_replacements_stay_on_device():
    device = torch.device("cuda")
    m = torch.nn.Module()
    m.keep = torch.nn.Parameter(torch.ones(4), requires_grad=False)  # CPU
    m.repacked = torch.nn.Parameter(torch.ones(4), requires_grad=False)  # CPU
    m.scale = torch.nn.Parameter(torch.ones(4), requires_grad=False)  # CPU
    m.resident = torch.nn.Parameter(torch.ones(4, device=device), requires_grad=False)
    slot = torch.full((2,), 7.0, device=device)
    with device_loading_context(m, device):
        assert m.keep.device.type == "cuda" and m.scale.device.type == "cuda"
        # An ordinary repack replacement keeps the loader's contract: it is
        # moved back to the CPU with the other CPU-resident parameters.
        replace_parameter(m, "repacked", torch.full((3,), 2.0, device=device))
        # A cache slot buffer is marked and stays where the kernel captured it.
        replace_parameter(m, "scale", slot)
        setattr(m.scale, DEVICE_RESIDENT_ATTR, True)
    assert m.keep.device.type == "cpu"
    assert m.repacked.device.type == "cpu" and m.repacked.shape == (3,)
    assert m.scale.device.type == "cuda" and m.scale.data_ptr() == slot.data_ptr()
    assert m.resident.device.type == "cuda"


def test_context_does_not_retain_moved_parameters():
    """Only names are recorded: a replaced-and-released weight is collectable
    while the context is still open."""
    import gc
    import weakref

    device = torch.device("cuda")
    m = torch.nn.Module()
    m.w = torch.nn.Parameter(torch.ones(4), requires_grad=False)  # CPU
    ref = weakref.ref(m.w)
    with device_loading_context(m, device):
        replace_parameter(m, "w", torch.zeros(4, device=device))
        gc.collect()
        assert ref() is None
