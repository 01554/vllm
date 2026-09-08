# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-model load scope for the expert tier.

Draft models (MTP, EAGLE) load through the same loader hooks as the target:
the native NVFP4 layout switch, the UVA expert offloader and the tier
registration in `process_weights_after_loading`. Those hooks are for the
target only, so the draft loader wraps its load in `draft_load_scope()` and
the hooks check `is_draft_load_scope()`. A ContextVar token keeps nesting
and exception exits correct; this module imports nothing from the tier
runtime so the hooks can query it before the runtime is imported.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

_DRAFT_LOAD: ContextVar[bool] = ContextVar("lab_expert_tier_draft_load", default=False)


@contextlib.contextmanager
def draft_load_scope() -> Iterator[None]:
    """Mark the enclosed model load as a draft (non-target) load."""
    token = _DRAFT_LOAD.set(True)
    try:
        yield
    finally:
        _DRAFT_LOAD.reset(token)


def is_draft_load_scope() -> bool:
    """Whether the current context is loading a draft model."""
    return _DRAFT_LOAD.get()


__all__ = ["draft_load_scope", "is_draft_load_scope"]
