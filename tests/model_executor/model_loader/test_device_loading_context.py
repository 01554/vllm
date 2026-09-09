# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""device_loading_context restores only the parameters it moved.

A quant method may replace a parameter under the same name while its weights
are being processed (the expert cache repoints scale parameters at device
slot buffers); the replacement must stay on the device, while an unreplaced
CPU-resident parameter is moved back to the CPU."""

import pytest
import torch

from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")


def test_replaced_parameter_is_not_pulled_back_to_cpu():
    device = torch.device("cuda")
    m = torch.nn.Module()
    m.keep = torch.nn.Parameter(torch.ones(4), requires_grad=False)  # CPU
    m.scale = torch.nn.Parameter(torch.ones(4), requires_grad=False)  # CPU
    m.resident = torch.nn.Parameter(torch.ones(4, device=device), requires_grad=False)
    slot = torch.full((2,), 7.0, device=device)
    with device_loading_context(m, device):
        assert m.keep.device.type == "cuda" and m.scale.device.type == "cuda"
        replace_parameter(m, "scale", slot)
    assert m.keep.device.type == "cpu"  # moved back, as before
    assert m.scale.device.type == "cuda" and m.scale.data_ptr() == slot.data_ptr()
    assert m.resident.device.type == "cuda"
