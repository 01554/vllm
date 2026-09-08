# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""N-gram (prompt-lookup) speculation for the MRv2 GPU model runner.

The MRv2 runner drives every speculator through `BaseSpeculator.propose`
after sampling, once `post_update` has appended the step's sampled tokens
to `RequestState.all_token_ids` and advanced `total_len`. N-gram drafting
needs exactly that history and nothing from the model, so this speculator
reuses the existing device n-gram kernel (`NgramProposerGPU`, no host
sync) on the request state the runner binds at startup:

- token history: `all_token_ids.gpu[idx_mapping]` (UVA-backed, fixed
  address) and `total_len.gpu[idx_mapping]`;
- eligibility: a request that sampled at least one token this step and
  holds at least `prompt_lookup_min` tokens;
- output: `[num_reqs, k]` int64 like the draft-model speculators. The
  runner schedules a fixed `k` drafts per request, so lanes the kernel
  leaves unmatched (-1) are filled with the request's last sampled token:
  a legal token that the verify step then rejects.

No draft logits (greedy rejection), no draft attention layers, no
multimodal inputs, nothing to capture.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


class NgramSpeculator(BaseSpeculator):
    supports_mm_inputs = False
    draft_logits = None
    draft_token_confidence_probs = None
    draft_attn_layer_names: frozenset[str] = frozenset()

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU

        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.vllm_config = vllm_config
        self.device = device
        self.k = int(vllm_config.num_speculative_tokens)
        self.min_n = int(speculative_config.prompt_lookup_min or 1)
        self.proposer = NgramProposerGPU(vllm_config, device)
        self._all_token_ids: torch.Tensor | None = None
        self._total_len: torch.Tensor | None = None

    def bind_request_states(self, req_states: Any) -> None:
        """Called by the runner once `RequestState` exists (fixed addresses)."""
        self._all_token_ids = req_states.all_token_ids.gpu
        self._total_len = req_states.total_len.gpu

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        return None

    def capture(self) -> None:
        return None

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        if self._all_token_ids is None or self._total_len is None:
            raise RuntimeError("NgramSpeculator was not bound to the request states")
        num_reqs = input_batch.num_reqs
        idx = input_batch.idx_mapping[:num_reqs].clamp(min=0).long()
        token_ids = self._all_token_ids.index_select(0, idx)
        num_tokens = self._total_len.index_select(0, idx).to(torch.int32)
        eligible = (num_sampled[:num_reqs] > 0) & (num_tokens >= self.min_n)
        return draft_from_history(
            self.proposer, token_ids, num_tokens, eligible, last_sampled, idx, self.k
        )


def draft_from_history(proposer, token_ids, num_tokens, eligible, last_sampled, idx, k):
    """Run the device n-gram kernel and fill unmatched lanes with a legal token."""
    with set_forward_context(None, proposer.vllm_config):
        drafts, _valid = proposer.kernel(num_tokens, token_ids, eligible)
    fill = last_sampled.index_select(0, idx).reshape(-1, 1).to(torch.int64)
    drafts = drafts.to(torch.int64)
    if drafts.shape[1] != k:
        raise RuntimeError("n-gram kernel width differs from num_speculative_tokens")
    return torch.where(drafts >= 0, drafts, fill.expand_as(drafts))


__all__ = ["NgramSpeculator", "draft_from_history"]
