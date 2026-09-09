# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build one MoE layer's raw NVFP4 bank straight from a safetensors shard.

Independent of the vLLM loader: the only transformation is concatenating
gate_proj and up_proj into the w13 bank (gate rows first) and stacking
experts. Global scales stay float32 scalars per projection here; the
backend's own float16 per-row expansion is recorded separately by the
bench, not applied by this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
FIELDS = ("weight", "weight_scale", "weight_scale_2", "input_scale")


def tensor_name(prefix: str, expert: int, proj: str, field: str) -> str:
    return f"{prefix}.experts.{expert}.{proj}.{field}"


def sha256_of(t: torch.Tensor) -> str:
    # reshape(-1) first: 0-dim scalars (the F32 global/input scales) cannot
    # be viewed as bytes directly.
    flat = t.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def load_layer_bank(
    shard: str | Path,
    prefix: str,
    num_experts: int,
    *,
    experts: list[int] | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Return (bank, manifest) for `experts` (default: all) of one layer.

    bank keys: w13_weight u8 [E, 2N, K/2], w13_weight_scale e4m3 [E, 2N, K/16],
    w13_weight_scale_2 f32 [E, 2] (gate, up), w2_weight u8 [E, K, N/2],
    w2_weight_scale e4m3 [E, K, N/16], w2_weight_scale_2 f32 [E].
    manifest: source tensor names, dtypes, shapes, sha256, input scales.
    """
    from safetensors import safe_open

    ids = list(range(num_experts)) if experts is None else list(experts)
    parts: dict[str, list[torch.Tensor]] = {
        k: []
        for k in (
            "w13_weight",
            "w13_weight_scale",
            "w13_weight_scale_2",
            "w2_weight",
            "w2_weight_scale",
            "w2_weight_scale_2",
        )
    }
    sources: dict[str, dict] = {}
    input_scales: dict[str, list[float]] = {p: [] for p in PROJECTIONS}
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        for e in ids:
            got = {}
            for proj in PROJECTIONS:
                for field in FIELDS:
                    name = tensor_name(prefix, e, proj, field)
                    t = f.get_tensor(name)
                    got[(proj, field)] = t
                    sources[name] = {
                        "dtype": str(t.dtype),
                        "shape": list(t.shape),
                        "sha256": sha256_of(t),
                    }
                input_scales[proj].append(float(got[(proj, "input_scale")]))
            parts["w13_weight"].append(
                torch.cat((got[("gate_proj", "weight")], got[("up_proj", "weight")]), 0)
            )
            parts["w13_weight_scale"].append(
                torch.cat(
                    (
                        got[("gate_proj", "weight_scale")],
                        got[("up_proj", "weight_scale")],
                    ),
                    0,
                )
            )
            parts["w13_weight_scale_2"].append(
                torch.stack(
                    (
                        got[("gate_proj", "weight_scale_2")].float(),
                        got[("up_proj", "weight_scale_2")].float(),
                    )
                )
            )
            parts["w2_weight"].append(got[("down_proj", "weight")])
            parts["w2_weight_scale"].append(got[("down_proj", "weight_scale")])
            parts["w2_weight_scale_2"].append(
                got[("down_proj", "weight_scale_2")].float()
            )
    bank = {k: torch.stack(v).contiguous() for k, v in parts.items()}
    manifest = {
        "shard": str(shard),
        "prefix": prefix,
        "experts": ids,
        "transformations": [
            "w13 = cat(gate_proj, up_proj) along rows (gate first)",
            "global scales kept as float32 scalars per projection",
        ],
        "bank": {
            k: {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": sha256_of(v)}
            for k, v in bank.items()
        },
        "sources": sources,
        "input_scales": input_scales,
    }
    return bank, manifest


def write_manifest(manifest: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
