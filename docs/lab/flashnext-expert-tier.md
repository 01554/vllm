# FlashNext expert tier: fork baseline and work plan

This is an experimental baseline for one draft PR in `01554/vllm`.
The objective is to serve FlashNext NVFP4 through vLLM within a 48 GiB VRAM
budget, reusing the existing RAM TEMP exchange and heat-based expert placement
policy. The adapter now follows a CUDA Graph contract (below) so decode can run
as FULL graph replays without torch.compile. **This has only been checked with
CPU unit tests; no GPU capture, replay, output, or speed result exists yet.**
It is awaiting human review.

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

## CUDA Graph contract

Removing `--enforce-eager` alone was insufficient: the eager adapter collected
routing on the CPU inside the last MoE layer, updated policy in Python,
performed conditional exchanges, and recreated device mapping tensors. The
graph-compatible adapter keeps the RAM TEMP storage contract and changes the
boundaries:

- **Fixed addresses.** Hot weights, pinned cold banks (read through UVA), both
  expert maps, and a static routing record buffer
  (`[max_tokens, 48 layers, 2k+1]` int32) are allocated once. Maps are
  republished in place after every exchange instead of being reallocated.
- **Inside the forward, layers only record.** Each MoE layer validates routing
  on the device and writes IDs, activity flags, and the padding mask into its
  row of the record buffer. No host copy, policy step, or exchange happens
  inside a layer, so the whole decode forward is capturable.
- **The runner finishes every forward.** `GPUModelRunner.execute_model` calls
  `finish_model_forward(model, num_tokens_after_padding)` after eager forwards
  and FULL replays alike. That hook performs the single D2H copy of the record
  prefix, the heat update, `plan_resync`, RAM TEMP swaps, in-place map
  publication, and waits for completion before the next forward is issued. A
  replay runs no Python in the layers, so the runner supplies the row count.
- **Startup bookkeeping.** Warmup and capture call the model directly and
  leave recorded forwards nobody finishes; before heat is enabled those are
  counted as `dropped_startup_records`. After heat is enabled an unfinished
  forward is an error. `captured_forwards`, `recorded_forwards`, and
  `replayed_forwards` are reported in `LAB_EXPERT_TIER_STATS`.
- **Batched exchanges.** A resync plan is executed in waves of
  slot-independent swaps (`plan_waves`): a wave issues every D2H hot→TEMP
  copy, waits once, every H2D cold→hot copy, waits once, then the host
  TEMP→cold writes. A swap that reuses a hot or cold slot touched earlier in
  the same plan starts a new wave, so the result is byte-identical to the
  sequential order the policy assumes. The pinned TEMP pool holds
  `VLLM_LAB_EXPERT_TIER_TEMP_SLOTS` rows (default 8), which also caps the
  wave size. `migration_waves` and `max_wave_swaps` are reported.
- **One reduction per layer.** With `VLLM_LAB_EXPERT_TIER_SPLIT=fused`
  (default) each MoE layer runs the hot and the cold Marlin GEMM chains
  directly into disjoint rows of one shared `[tokens*k, hidden]` workspace
  buffer and reduces it once. Every routing slot belongs to exactly one
  partition, and padding slots stay at the zero fill. This removes both
  output allocations, both masked reductions, the hot clone, the add, and
  the modular-kernel prepare/finalize wrappers per layer; one block
  alignment and two GEMMs per partition remain. `SPLIT=modular` restores the
  two stock modular kernel calls. Init verification compares either path
  against the source kernel.
- **Staged cold experts for batch-1 decode** (`VLLM_LAB_EXPERT_TIER_STAGING`,
  default 0; opt-in). Measured correct on the GPU (staged init checks, FULL
  replay, 48 GiB audit) but 7.74% slower than the fused two-partition path
  in its first A→B (B 30.49 vs 33.05 tok/s), so it stays off until the
  per-layer plan and copy launches are fused. Each layer's hot bank has `top_k` spare rows, charged to the
  capacity budget (32 GiB: 258 hot slots become 248 plus 10 staging rows).
  For a one-token forward, `plan_staging` derives on the device, with fixed
  shapes and no host sync, the distinct selected cold experts, a gather index
  per spare row, the staged count, and a per-step expert map rebuilt from the
  hot map. `gather_staging` copies exactly that many rows of all six tensors
  from the pinned cold bank into the spare rows (a Triton kernel reads the
  count on the device and exits early for unused slots), and hot plus staged
  rows run through one Marlin chain from VRAM. Prefill and any forward wider
  than one token keep the fused two-partition path. Staged rows are
  transient copies, never written back, and outside the swap slot range.
  Init verification adds single-row hot, cold, mixed, and padding checks
  through this path. `vllm/_lab_expert_tier/staging.py` holds the pure
  functions and their CPU contract tests.
- **Observer seam** (`VLLM_LAB_EXPERT_TIER_OBSERVER`, default `records`). The
  per-forward routing record and its readback live in an observer object;
  `device` selects `heat_device.DeviceObserver` (separate work) which keeps
  heat on the device and hands the coordinator a `DeviceSnapshot` at its own
  cadence, or `Deferred` for forwards with no readback. The coordinator
  counts forwards once per finish, plans only on a snapshot, and `flush`
  collects deferred observations without planning unless asked.
- **Supported modes.** Compilation mode must be NONE (no torch.compile), and
  the cudagraph mode must be NONE, FULL_DECODE_ONLY, or FULL. Piecewise
  cudagraphs and `VLLM_USE_BREAKABLE_CUDAGRAPH` are rejected.

To enable decode graphs, replace `--enforce-eager` in the lab launcher with:

```bash
--compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'
```

`--enforce-eager` still works and reproduces the eager path through the same
hook. The `cuda_graphs` value in `LAB_EXPERT_TIER_READY` is the configured
mode at weight-loading time; MRv2 resolves the final mode later (it may
downgrade FULL_DECODE_ONLY to NONE if the attention backend lacks full-graph
support). Judge graph use from the capture log, `captured_forwards` in the
heat-enable `startup_stats`, and a growing `replayed_forwards` during decode,
not from the READY line.

GPU validation that remains to be done, in order:

1. Startup with the flag above: the capture log shows FULL graphs captured,
   `startup_stats.captured_forwards` is nonzero when heat is enabled, init
   verification passes, and the 48 GiB capacity audit holds with the graph
   pool allocated.
2. A→B with heat enabled: `replayed_forwards` grows with decode, `swaps` and
   `resyncs` are nonzero, and no poisoned-tier error appears. Compare the B
   output with the eager run of the same source revision; the graph path
   should not change routing or expert arithmetic, but this is unverified.
3. Decode tokens/s versus the eager tier (11.1117) and the fixed baseline
   (7.4182), measured without the profiler.
4. A profile run to confirm per-decode CPU launch count drops from ~3,300 and
   to attribute what remains (host unpack, policy loops, exchange copies).

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
