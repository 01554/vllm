# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""moe_expert_cache_provider: config field, CLI -> EngineArgs, hash."""

import pytest
from pydantic import ValidationError

from vllm.config.offload import OffloadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def test_default_is_the_upstream_cached_provider():
    assert OffloadConfig().moe_expert_cache_provider == "cached"


def test_hash_distinguishes_the_provider():
    cached = OffloadConfig(moe_expert_cache_size=8).compute_hash()
    row = OffloadConfig(
        moe_expert_cache_size=8, moe_expert_cache_provider="row"
    ).compute_hash()
    assert cached != row


def test_invalid_provider_is_rejected():
    with pytest.raises(ValidationError):
        OffloadConfig(moe_expert_cache_provider="lru")  # type: ignore[arg-type]


def test_cli_reaches_engine_args_and_offload_config():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        ["--moe-expert-cache-size", "16", "--moe-expert-cache-provider", "row"]
    )
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.moe_expert_cache_provider == "row"
    # The same kwargs EngineArgs.create_engine_config passes on.
    offload = OffloadConfig(
        moe_expert_cache_size=engine_args.moe_expert_cache_size,
        moe_expert_cache_split=engine_args.moe_expert_cache_split,
        moe_expert_cache_provider=engine_args.moe_expert_cache_provider,
    )
    assert (offload.moe_expert_cache_size, offload.moe_expert_cache_provider) == (
        16,
        "row",
    )
