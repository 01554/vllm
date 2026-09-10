# Small decode alignment experiment

Fork-side experiment based on `205983f839f5155c0ebd066dd84e77f3ec41bc84`.
Not part of upstream PR #56177. Do not combine with vector staging or other
optimizations for the first comparison.

The candidate replaces logical alignment plus physical remap only for decode
with 1..64 lanes and bank rows > logical experts. Wide batches retain the
baseline path. Tables, validation, safe IDs, copy ordering, output zeroing and
activation are unchanged. Same-expert route order is stable lane order; the
reference atomic ordering need not be byte-identical.

## Correctness before timing

On the GPU environment, with the candidate Python source installed:

```sh
.venv/bin/python -m pytest tests/kernels/expert_pool -v
.venv/bin/python benchmarks/kernels/benchmark_pool_small_align.py --output /results/align-bench
```

The existing consumer suite covers promotion/eviction, cross-layer victims,
wide/decode transitions, safe IDs, clamp activation, graph replay and isolated
device assertions. The added alignment test covers 1/10/64 lanes and every
Marlin block size, all padding, duplicate routes, absent experts, and physical
rows >= E. It compares route groups and padded length against the baseline
CUDA align + P1 remap, and checks initialized output tails after each replay.

Timing uses 30 batches of 100 graph replays, alternates order each batch, retains
both inputs/outputs/graphs and saves all samples. It measures alignment alone;
planner, copies, consumer, PLE and serving overhead are excluded. PTX/cubin and
register/spill counts for the candidate are saved. SASS can be extracted with
`cuobjdump --dump-sass <file.cubin>` where the tool is installed; a missing
disassembler is not a correctness result. Preserve source archive/head and
environment provenance alongside results. GPU execution belongs to Astra.

Static lint and Python compilation passed locally. CUDA correctness, compiled
kernel resources, sanitizer behavior and performance remain unverified until
the GPU run. In particular the single-CTA initialize/barrier/scatter sequence
requires review. Do not infer serving improvement from kernel duration sums.
Any subsequent serving comparison keeps the existing 5% adoption and FT95%
criteria, with fresh runs and unchanged inputs/configuration.

The runner replays both retained graphs before and after timing, checking route
groups, per-group padded block counts, and output counts against the input.
Candidate unused tails are checked as well. A failure stops the run and writes
`invalid.json` with `timing_valid=false`; no final timing report is produced.
