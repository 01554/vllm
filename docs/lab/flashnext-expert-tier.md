# FlashNext expert tier: fork baseline and work plan

This is an experimental baseline for one draft PR in `01554/vllm`.
The objective is to serve FlashNext NVFP4 through vLLM within a 48 GiB VRAM
budget, reusing the existing RAM TEMP exchange and heat-based expert placement
policy. CUDA Graph support is **not implemented** in this baseline. It currently
requires eager execution and is awaiting human review.

## Baseline and provenance

- vLLM base: `1970f3ed4`.
- PLE support: 10 runtime files imported from upstream PR `#54129`.
- Expert tier: the existing adapter and policy under `vllm/_lab_expert_tier/`.
  Hot experts reside in VRAM; cold experts reside in pinned RAM and are accessed
  through GPU aliases. Periodic heat-based exchanges use RAM TEMP storage.
- FreeToken is a reference implementation only. No FreeToken source is copied
  into this baseline. The reference checkout is pinned to
  `af71ba43206e124f5ff6419b47ee36c6e9981078`; its LRU dependency is
  `flashlib==0.3.0`.
- The policy's existing MIT provenance is retained in
  `vllm/_lab_expert_tier/LICENSE.llama-cpp`.

Upstream PR `#37190` already covers related work. This draft is for review and
integration inside the fork; it is not a duplicate upstream PR submission.
AI assistance is used for this work. A human must review the changes and
understand their behavior before approval.

## One PR, three roles

| Role | Branch | Responsibility |
| --- | --- | --- |
| CUDA Graph implementation | `lab/flashnext-cuda-graph` | Make buffer lifetimes, replay boundaries, and cache updates compatible with graphs. |
| CPU and FreeToken comparison | `lab/flashnext-cpu-compare` | Map host work, CUDA launches, synchronization, and kernel paths to source and existing traces. |
| GPU validation and integration | `lab/flashnext-integration` | Integrate changes, run controlled comparisons, and maintain the single draft PR. |

The PR base `lab/flashnext-base` pins unmodified vLLM at `1970f3ed4`.
Both task branches start at the same imported-baseline commit on
`lab/flashnext-integration` and use separate worktrees. Coordinate ownership before editing the same files; the
comparison role supplies findings and validation requests to the implementation
role rather than changing its runtime concurrently.

**Only `astra_vllm_freetoken` may operate the shared RTX PRO 6000 GPU.** This
includes model servers, GPU tests and profilers, Docker workloads, and the VRAM
balloon. Other roles use source inspection and CPU-only analysis and send
concrete experiment requests to the GPU owner.

## Historical measurements

These are previous lab runs, not measurements of this source import. The model
has **not been rerun for this import**.

| Configuration | Decode tokens/s |
| --- | ---: |
| Fixed-placement vLLM baseline, eager | 7.4182 |
| Existing expert tier, best measured settings, eager | 11.1117 |
| Stock FreeToken, CUDA Graph ON | 77.4258 |
| Stock FreeToken, CUDA Graph OFF | 20.8291 |

The test GPU was a 96 GB-class RTX PRO 6000 Blackwell with CUDA-reported free
VRAM constrained to **48 GiB** by a balloon. This constrains capacity; it does
not reproduce Ada compute throughput or bandwidth. **RTX 6000 Ada hardware
has not been validated.**

The workload uses a fixed SWELancer prompt A for warmup, then prompt B for
measurement, at concurrency 1. Decode throughput is the client-observed
`(completion_tokens - 1) / (last_text_time - first_text_time)`, excluding
prefill. The prompts are inference workloads, not graded SWELancer solutions.
FreeToken ON/OFF produced identical text; vLLM and FreeToken outputs differ.
The FreeToken graph flag also changes related PLE host staging, so its ratio
does not isolate CUDA launch cost alone or predict the vLLM graph speedup.

Evidence is retained in the separate lab workspace. The following are paths
relative to that workspace, **not files included in this fork**:

- `results/swelancer-tier-20260907/comparison-phase1.json`
- `results/swelancer-tier-sweep-20260907-remaining/comparison.json`
- `results/swelancer-tier-sweep-20260907-remaining/capacity-audit-all.json`
- `results/swelancer-tier-v3-build-20260907/source/`
- `results/freetoken-reference-20260907-graph-on/benchmark.jsonl`
- `results/freetoken-reference-20260907-graph-off/benchmark.jsonl`
- `results/freetoken-reference-20260907-graph-off/graph-on-off-comparison.json`
- `results/freetoken-reference-20260907-profile/trace-analysis.json`
- `results/freetoken-reference-20260907-profile/vllm-comparison.json`
- `notes/freetoken-reference-profile-2026-09-07.md`

## Validation and next change

From the worktree root, use the prepared virtual environment for CPU tests:

```bash
.venv/bin/python -m unittest discover -s tests/lab_expert_tier -p 'test_*.py' -v
```

This command does not establish model correctness or GPU performance. Record
its actual result with the integration revision; do not carry historical test
passes forward as evidence for changed code.

The first implementation task is graph compatibility. Removing
`--enforce-eager` alone is insufficient: the current adapter collects routing
on the CPU, updates policy in Python, performs conditional exchanges, and
recreates device mapping tensors. Graph work must define persistent buffers,
replay-external policy updates, and ordering that keeps weights and maps
consistent while preserving the existing RAM TEMP storage contract.

The comparison role should identify remaining host and kernel differences
without assuming that all of the gap is CPU computation. GPU validation must
check output consistency, changing routes and placements across replay,
the 48 GiB capacity constraint, and normal decode speed separately from
profiler runs. Graph performance and Ada behavior remain unverified.

## GPU worktree image

The integration owner can build Python-only changes on the fixed native wheel:

```bash
test -z "$(git status --porcelain)" || { echo "Use a clean worktree"; exit 1; }
revision=$(git rev-parse HEAD)
docker build -f docker/Dockerfile.flashnext-lab \
  --build-arg LAB_REVISION="$revision" \
  -t "lab/vllm-flashnext:worktree-${revision:0:12}" .
```

The recipe merges the checked-out Python sources into the pinned wheel and
retains its native extensions. Do not use it for C++/CUDA extension changes or
a different upstream base. This recipe has not been built or model-tested in
the workspace setup step. The existing lab launcher accepts the selected image
via `FLASHNEXT_TIER_IMAGE`; it does not automatically use the worktree.

Hold the lab's GPU lock for the complete server, balloon, warmup, measurement,
and cleanup lifecycle. A detached server outlives its launcher, so locking only
`docker run -d` does not reserve the GPU for an experiment.
