# P1 vector staging and small alignment bundle

Fork-only independent experiment based on P1
`205983f839f5155c0ebd066dd84e77f3ec41bc84`. Not part of upstream #56177.

The component patches are retained without production modifications:

- Vector staging: `aad8d8f48457e745798e2e3528128ddaa05f8b9e`.
- Small alignment: `f40a44797f1b8796d9a1cdfe5125eed10d0acf59`.

Production changes are confined to tables.py's two masked suffix loops,
layer.py's small decode alignment dispatch, and small_align.py. Validation,
safe_ids, sticky error, ownership planning, copies, output zeroing, activation,
wide fallback, and PLE retain P1 behavior. The vector suffix publishes step_map
before the existing barrier; alignment consumes that map after planning/copy
ordering. Combining these independently checked components still requires new
GPU regression and serving evidence.

## GPU recipe (Astra owns execution)

Use a fresh result directory and the established full environment. Preserve the
fixed source archive, baseline source, environment, logs, compiler artifacts and
all raw timing samples. From the candidate checkout:

```sh
git show 205983f839f5155c0ebd066dd84e77f3ec41bc84:vllm/model_executor/layers/fused_moe/expert_pool/tables.py > /results/baseline-tables.py
.venv/bin/python -m pytest -q tests/kernels/expert_pool tests/config/test_moe_expert_pool_rows.py
.venv/bin/python benchmarks/kernels/expert_pool/bench_staging.py \
  --check-weights --baseline /results/baseline-tables.py \
  --candidate vllm/model_executor/layers/fused_moe/expert_pool/tables.py \
  --output /results/staging
.venv/bin/python benchmarks/kernels/benchmark_pool_small_align.py \
  --output /results/alignment
```

Keep isolated assertion subprocess logs. Both local runners retain their
correctness checks and alternating 30 x 100 graph timing. The full pytest suite
combines the existing consumer/graph/clamp tests, staging state comparisons and
small alignment cases. Stop on any failure before interpreting timing. Missing
SASS is recorded as missing.

## Predeclared serving decision

After regression, compare fresh A-to-B requests three times, without profiling,
against P1 at the unchanged N258/4096/48GiB recipe. Preserve raw SSE, request and
token-ID hashes, timing, capacity samples, and owned-container final state.
P1 median is 69.10308968115035 tok/s; the 5% adoption threshold is
72.55824416520787. The matched FreeToken 95% threshold is 72.6129502454466.

Neither component met standalone adoption: vector median 70.45922485261384,
small alignment median 71.70555673559706. Their measurements are historical,
with different generated trajectories. They neither predict nor establish
bundle improvement. Bundle GPU correctness and E2E performance are untested.
