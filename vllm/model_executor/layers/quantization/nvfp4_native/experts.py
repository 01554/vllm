# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-resident NVFP4 MoE experts on the raw ModelOpt bank.

Selected explicitly with ``moe_backend="native"``; never auto-selected.
Decode rows (``M <= gemv_rows``) run the Triton GEMV adapter, larger
batches run the grouped prefill kernel.  Activations stay BF16 (no input
scales), router weights are applied once after the down projection and
routes are reduced inside the kernels, so the modular kernel's finalize
step is a no-op.  All large scratch comes from the modular kernel's
workspace (the worker's WorkspaceManager, separated per execution lane);
the experts keep only the bank views, an identity row map and a sticky
error flag.
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
        self._error: torch.Tensor | None = None
        self._rows = 0
        self._hidden = 0
        self._intermediate = 0

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
        rows, hidden, intermediate = native_bank.validate_bank(bank)
        device = bank["w13_weight"].device
        self._bank = bank
        self._rows, self._hidden, self._intermediate = rows, hidden, intermediate
        self._step_map = torch.arange(rows, dtype=torch.int32, device=device)
        self._error = torch.zeros(1, dtype=torch.int32, device=device)
        if device.type == "cuda":
            from . import kernels, prefill_kernels

            kernels.warmup(device)
            prefill_kernels.warmup(device)

    def _scratch_nbytes(self, max_tokens: int, top_k: int) -> int:
        decode = native_bank.scratch_nbytes(
            native_bank.decode_scratch_layout(
                self._hidden, self._intermediate, min(max_tokens, self.gemv_rows), top_k
            )
        )
        prefill = native_bank.scratch_nbytes(
            native_prefill.prefill_scratch_layout(
                self._hidden, self._intermediate, max_tokens, top_k, self._rows
            )
        )
        return max(decode, prefill)

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
        # workspace1 only carries the output (M, K); workspace2 is the flat
        # kernel scratch, declared in elements of the workspace dtype (the
        # activation dtype, 2 bytes) and carved as bytes in apply().
        if self._bank is None:
            raise RuntimeError("NativeNvFp4Experts: weights were not processed")
        nbytes = self._scratch_nbytes(M, topk)
        elems = (nbytes + 1) // 2
        return ((M, K), (elems,), (M, K))

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
        top_k = ids.shape[1]
        rows = hidden_states.shape[0]
        assert self._error is not None
        scratch = workspace2.reshape(-1).view(torch.uint8)
        if rows <= self.gemv_rows:
            workspace = native_bank.carve_workspace(
                self._bank, scratch, rows, top_k, self._error, num_experts=self._rows
            )

            result = native_bank.gemv(
                hidden_states,
                weights,
                ids,
                self._bank,
                self._step_map,
                workspace,
                activation=act,
            )
        else:
            workspace = native_prefill.carve_workspace(
                self._bank, scratch, rows, top_k, self._error, num_experts=self._rows
            )
            result = native_prefill.prefill(
                hidden_states,
                weights,
                ids,
                self._bank,
                self._step_map,
                workspace,
                activation=act,
            )
        output.copy_(result)
