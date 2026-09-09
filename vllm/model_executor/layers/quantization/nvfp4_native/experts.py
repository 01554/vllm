# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-resident NVFP4 MoE experts on the raw ModelOpt bank.

Selected explicitly with ``moe_backend="native"``; never auto-selected.
Decode rows (``M <= gemv_rows``) run the Triton GEMV adapter, larger
batches run the grouped prefill kernel.  Activations stay BF16 (no input
scales), router weights are applied once after the down projection and
routes are reduced inside the kernels, so the modular kernel's finalize
step is a no-op.
"""

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Static,
)
from vllm.platforms import current_platform

from . import bank as native_bank
from . import prefill as native_prefill
from .bank import BANK_TENSORS
from .loader import activation_name

# Rows at or below this count take the decode GEMV; above it, grouped prefill.
DEFAULT_GEMV_ROWS = 1

# Scratch is shared by every layer with the same bank shape and capacities:
# layers run sequentially on one stream, and per-layer copies would cost
# hundreds of MiB each at prefill batch sizes (e.g. ~390 MiB per layer at
# 4096 tokens x top-10 on a 2560/640 model, ~18 GiB over 48 layers).
_DECODE_WORKSPACES: dict[tuple, native_bank.Workspace] = {}
_PREFILL_WORKSPACES: dict[tuple, native_prefill.Workspace] = {}


def _workspace_key(bank: native_bank.Bank, max_tokens: int, top_k: int) -> tuple:
    rows, hidden, intermediate = native_bank.validate_bank(bank)
    device = bank["w13_weight"].device
    return (str(device), rows, hidden, intermediate, max_tokens, top_k)


def shared_decode_workspace(
    bank: native_bank.Bank, max_tokens: int, top_k: int
) -> native_bank.Workspace:
    key = _workspace_key(bank, max_tokens, top_k)
    workspace = _DECODE_WORKSPACES.get(key)
    if workspace is None:
        workspace = native_bank.allocate_workspace(
            bank, max_tokens, top_k, num_experts=key[1]
        )
        _DECODE_WORKSPACES[key] = workspace
    return workspace


def shared_prefill_workspace(
    bank: native_bank.Bank, max_tokens: int, top_k: int
) -> native_prefill.Workspace:
    key = _workspace_key(bank, max_tokens, top_k)
    workspace = _PREFILL_WORKSPACES.get(key)
    if workspace is None:
        workspace = native_prefill.allocate_workspace(
            bank, max_tokens, top_k, num_experts=key[1]
        )
        _PREFILL_WORKSPACES[key] = workspace
    return workspace


class NativeNvFp4Experts(mk.FusedMoEExpertsModular):
    """Raw-layout NVFP4 experts (no repack) for ModelOpt checkpoints."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        gemv_rows: int = DEFAULT_GEMV_ROWS,
    ):
        super().__init__(moe_config, quant_config)
        if gemv_rows < 1:
            raise ValueError("gemv_rows must be positive")
        problem = self._unsupported_reason(moe_config)
        if problem is not None:
            raise ValueError(f"native NVFP4 experts: {problem}")
        self.gemv_rows = gemv_rows
        self._bank: native_bank.Bank | None = None
        self._step_map: torch.Tensor | None = None
        self._decode_workspace: native_bank.Workspace | None = None
        self._prefill_workspace: native_prefill.Workspace | None = None

    # --- capability statics -------------------------------------------------

    @staticmethod
    def _unsupported_reason(moe_config: FusedMoEConfig) -> str | None:
        """Configurations the kernels cannot honour; rejected explicitly
        rather than silently computing something else."""
        if moe_config.in_dtype != torch.bfloat16:
            return f"activations must be bfloat16, got {moe_config.in_dtype}"
        if getattr(moe_config, "is_lora_enabled", False):
            return "LoRA is not supported"
        if getattr(moe_config, "has_bias", False):
            return "expert bias is not supported"
        for name in ("swiglu_limit", "swiglu_alpha", "swiglu_beta"):
            if getattr(moe_config, name, None) is not None:
                return f"{name} is not supported (plain SiLU only)"
        return None

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        problem = NativeNvFp4Experts._unsupported_reason(moe_config)
        if problem is not None:
            return False, f"kernel does not support {problem}"
        return mk.FusedMoEExpertsModular.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        # NVFP4 weights with 16-bit activations only.
        return weight_key == kNvfp4Static and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # Every expert row is resident on this device; no expert parallelism.
        return not moe_parallel_config.use_ep

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return None

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    # --- weights ------------------------------------------------------------

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"):
            if getattr(self.quant_config, name, None) is not None:
                raise ValueError(
                    f"native NVFP4 experts: {name} is not supported (plain SiLU only)"
                )
        bank = {name: getattr(layer, name).data for name in BANK_TENSORS}
        rows, _hidden, _intermediate = native_bank.validate_bank(bank)
        top_k = self.moe_config.experts_per_token
        self._bank = bank
        self._step_map = torch.arange(
            rows, dtype=torch.int32, device=bank["w13_weight"].device
        )
        self._decode_workspace = shared_decode_workspace(bank, self.gemv_rows, top_k)
        self._prefill_workspace = shared_prefill_workspace(
            bank, self.moe_config.max_num_tokens, top_k
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # The kernels own their scratch; the modular kernel only needs the
        # output buffer, which it provisions from workspace1.
        return ((M, K), (1,), (M, K))

    # --- forward ------------------------------------------------------------

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if self._bank is None or self._step_map is None:
            raise RuntimeError("NativeNvFp4Experts: weights were not processed")
        if expert_map is not None:
            raise NotImplementedError("native NVFP4 experts do not support EP")
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "native NVFP4 experts apply router weights after the down projection"
            )
        act = activation_name(activation)
        ids = topk_ids.to(torch.int32).contiguous()
        # The adapters require FP32 router weights.
        weights = topk_weights.to(torch.float32).contiguous()
        if hidden_states.shape[0] <= self.gemv_rows:
            assert self._decode_workspace is not None
            result = native_bank.gemv(
                hidden_states,
                weights,
                ids,
                self._bank,
                self._step_map,
                self._decode_workspace,
                activation=act,
            )
        else:
            assert self._prefill_workspace is not None
            result = native_prefill.prefill(
                hidden_states,
                weights,
                ids,
                self._bank,
                self._step_map,
                self._prefill_workspace,
                activation=act,
            )
        output.copy_(result)
