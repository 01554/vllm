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
- **Asynchronous exchanges** (`VLLM_LAB_EXPERT_TIER_ASYNC_MIGRATION`, default
  0; opt-in). Each layer gains `TEMP_SLOTS` spare VRAM rows after the staging
  rows and `TEMP_SLOTS` spare pinned RAM rows after the cold rows, both
  charged to the budgets (32 GiB, staging on: 258 hot slots become 240).
  The policy's logical slots are unchanged; each layer maps them to physical
  rows, and the device maps publish physical rows. A resync plan runs
  asynchronously only when every layer's swaps fit its free spare rows and
  no hot or cold slot repeats within a layer; otherwise the whole plan takes
  the synchronous path in its original order (`sync_fallbacks`,
  `fallback_over_budget`, `fallback_slot_reuse`). An eligible plan is
  queued on a migration stream after the compute stream: promoted cold rows
  go to spare VRAM rows and evicted hot rows to spare RAM rows while the
  next forward still reads the old placement. At the next boundary, before
  any observation or plan, the coordinator waits for the transfer if it is
  not done, makes the compute stream wait on it, records a retire fence,
  flips the row tables, republishes the maps, commits the policy, and
  rebases the observer; retired rows return to their rings behind that
  fence. So the first version overlaps at most one forward and then waits;
  placement takes effect one forward after the plan. Outputs use the same
  mathematical weights, but bit-level equivalence is not claimed.
  `async_plans`, `async_commits`, `async_wait_seconds`, and
  `async_pending_boundaries` are reported.
- **Per-layer hot capacity** (`VLLM_LAB_EXPERT_TIER_LAYER_SLOTS`, default
  `uniform`). Either the uniform split or 48 comma-separated hot slot counts,
  one per layer. Load-time checks: every layer keeps a hot and a cold
  partition, both at least `top_k` rows when init verification is on, and
  the exact per-layer byte sum (hot + staging + spare rows) fits the
  capacity; nothing is rescaled silently. Bank sizes are fixed at load, so
  captured graph addresses are unaffected. The policy takes the per-layer
  tuple (`hot_slots_per_layer`); `LAB_EXPERT_TIER_READY` lists per-layer
  counts when they differ. This makes capacity configurable per layer; it
  is not an automatic allocation and not FreeToken's shared demand-driven
  pool.
- **Promote mode** (`VLLM_LAB_EXPERT_TIER_PROMOTE`, default 0; needs
  `STAGING=1` and `ASYNC_MIGRATION=0`). Per-token cache management on the
  device in the FreeToken style: a device LRU planner (`PLANNER=device`,
  `vllm/_lab_expert_tier/device_lru.py`, separate ownership; `reference` is
  the torch semantics for tests) decides each layer step which selected
  cold experts are promoted, which unselected hot experts are evicted
  (least recent, tie by logical hot slot), and which misses are staged only;
  the flip (`promote.py`) updates the device tables first (logical owner
  maps, logical-to-physical rows, a free VRAM ring of `TEMP_SLOTS` rows, a
  RAM pool of `TEMP_SLOTS` unreferenced rows with shadow tags) and writes
  the copy lists and the step map; then one launch gathers promoted and
  staged rows from RAM, one evicts victims to RAM, and the bank kernel runs
  through the step map. The actual order is flip → gather → evict → MoE:
  the tables describe the placement the rest of the step will produce, and
  nothing reads them in between because all of it is queued on the compute
  stream and the host only reads the tables at forward boundaries (stats
  reports). Map publication is therefore not "after transfer completion"
  in a host sense; correctness rests on stream order, and any failure
  poisons the tier. Ownership: hot and cold owner maps stay exclusive; a
  hot expert may keep a valid RAM copy (its shadow), reclaimed without a
  copy if it is evicted while the shadow is intact. The heat policy only
  observes; the host maps are refreshed from the device at each stats
  report, where the tables are validated. Startup and init verification
  keep the gate closed (everything staged only); `enable_heat` opens it.
  This relaxes the exclusive placement deliberately (RAM copies allowed
  within the fixed RAM footprint) and is not verified on a GPU.
- **RAM backing** (`VLLM_LAB_EXPERT_TIER_RAM_BACKING`, default 0; needs
  `PROMOTE=1`). The loader's pinned UVA source of every expert (row =
  expert id, the same allocation the loader already holds at READY) is
  kept as the RAM bank instead of compacting the cold rows into a new
  pinned allocation, so RAM row `e` holds expert `e` for the life of the
  process, as in FreeToken. An eviction then only flips the tables: no
  D2H copy, no RAM pool, no shadows (`ram_free` is a placeholder of the
  VRAM ring's length so the planner capacity stays the ring length, and
  `ram_shadow` is the identity). The evict launch is skipped in
  `split_promote`, and the init-verification swap copies in only. Host
  bytes: the full expert source stays resident (63.3 GiB for 48 x 512
  experts, the loader's own peak) instead of the cold rows plus pool;
  no bank is held twice. VRAM budget and addresses are unchanged. The
  READY log carries `host_bytes` and `ram_backing`.
- **Global pool** (`VLLM_LAB_EXPERT_TIER_GLOBAL_POOL`, default 0; needs
  `RAM_BACKING=1`, hence `PROMOTE=1` and `STAGING=1`). One VRAM bank
  shared by every layer (`global_pool.py`), FreeToken style: rows are
  keyed by (layer, expert), one LRU over all keys, and the resident count
  per layer floats (uniform or `LAYER_SLOTS` is only the starting
  placement). Each layer's batch-1 step is one Triton program: distinct
  selections in first-occurrence order are stamped with the step clock;
  each miss takes the row of the least recently used resident expert of
  any layer (ties by key) and is copied in from this layer's RAM bank
  (the victim's RAM row is intact, nothing is copied out); a miss with no
  victim, or any miss while the gate is closed, is staged only into
  `top_k` staging rows shared by all layers. The step map is the layer's
  slice of the pool's `hot_phys`, so the eager two-partition path reads
  the same slices (hot rows in the pool, cold rows in the layer's RAM
  bank). The pool's staging rows are charged once; everything else in the
  budget is pool rows, so the per-layer staging and spare rows of promote
  mode are freed for residency. Host swaps (init verification) go through
  `GlobalPool.host_swap` while the gate is closed. Stats reports validate
  the pool (`check_global_tables`) and log `pool_resident_per_layer`.
  Not verified on a GPU.
- **Native backend** (`VLLM_LAB_EXPERT_TIER_MOE_KERNEL=native`, default
  `marlin`; needs `RAM_BACKING=1` and `SPLIT=fused`). The FreeToken-derived
  NVFP4 GEMV adapter (`native_nvfp4.py`, separate ownership) replaces the
  Marlin chains. The loader branch (`native_loader.py`, called at the top
  of ModelOpt's `process_weights_after_loading`) keeps the checkpoint
  layout for every expert bank: packed uint8 `[E, 2I, H/2]` / `[E, H, I/2]`,
  E4M3 block scales, and per-row float16 globals expanded from the
  checkpoint's `[E, 2]` / `[E]` (gate rows take column 0, up rows column
  1; Marlin folds these into one). Input scales are dropped (BF16
  activations), no Marlin kernel object is built, and the layout is
  exclusive per process. The runtime calls `gemv(x, weights, ids, bank,
  map, workspace)` once per partition: the decode paths pass their step
  map; the eager two-partition path masks each partition's foreign routes
  to padding so the adapter never records them as missing. Workspaces are
  allocated once per physical row count (the bank and the RAM source) and
  shared by every layer, sized by the runner's token budget. Init
  verification compares the tier against the adapter over the full RAM
  source (every expert's own row, identity map). The adapter's sticky
  routing error is read at each stats report and poisons the tier. SiLU
  only. Measured with the global pool: B 48.92 tok/s. Multi-token rows
  run the decode GEMV per route by default (slow prefill);
  `VLLM_LAB_EXPERT_TIER_NATIVE_PREFILL=grouped` calls
  `native_prefill.prefill` (separate ownership) with the same arguments
  and that module's own workspace (allocated at init per physical row
  count) for rows > 1; decode is unaffected either way.
- **Fused shared-expert gate** (`VLLM_LAB_EXPERT_TIER_SHARED_GATE=fused`,
  default `torch`). The Qwen4 exp shared expert's gate,
  `sigmoid(x @ w) * out`, is a cuBLAS dot (two kernels), a sigmoid, and a
  broadcast multiply per layer in vLLM; FreeToken computes the gate in one
  Triton program and applies it in another. `shared_gate.py` does the dot,
  the sigmoid, and the scaling in one program per token, keeping vLLM's
  rounding points (dot, sigmoid, and product each round to the activation
  dtype). The hook is in `Qwen4ExpSparseMoeBlock.__init__` and replaces
  the shared MLP's forward; the routed + shared addition stays in the MoE
  runner. CPU test holds the reference equal to the module's own forward.
  Not verified on a GPU.
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
