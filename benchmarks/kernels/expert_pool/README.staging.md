# Staged suffix vectorization experiment

Baseline: `5119ca4906a1030305be0c52f6536288deba7c51` (#56177).
The change vectorizes two compacted staging suffix loops. It preserves masks,
barriers, promotion order, and ownership. It is not unconditional GPU memory
access, nor evidence that CPU branch-prediction results transfer to GPUs.

In the full CUDA environment, from the candidate checkout:

```sh
git show 5119ca4906a1030305be0c52f6536288deba7c51:vllm/model_executor/layers/fused_moe/expert_pool/tables.py > /tmp/pool-staging-baseline.py
python -m pytest -q tests/kernels/expert_pool
python benchmarks/kernels/expert_pool/bench_staging.py \
  --baseline /tmp/pool-staging-baseline.py \
  --candidate vllm/model_executor/layers/fused_moe/expert_pool/tables.py \
  --output /tmp/pool-staging-comparison
```

Use the environment's Python, not system Python. Preserve the whole fresh output
folder, command/environment, test log, and candidate commit. The runner saves
Triton compiler artifacts (including PTX and cubin where exposed), registers,
spills, SHA256 of both implementations, correctness results and 30 raw alternating
batches of 100 graph replays. If `cuobjdump` is available, disassemble each saved
cubin with `cuobjdump --dump-sass <file>` and save the output alongside it. Missing
SASS must be reported as missing, not inferred from source or PTX.

The WIDTH=16 and 64 measurements use 48 layers, 512 experts, 258 resident rows per
layer; staging capacity equals WIDTH. They isolate the planner and exclude expert
copies/consumer/PLE. These staging capacities are synthetic, not an exact serving
launch recipe. All-hit is gate-open; staged conditions are gate-closed so the
working set does not silently become resident during timing. Promotions are
correctness-only. No serving speedup can be claimed from these results alone.

Compare medians and raw batch variation; check all-hit regressions, register and
spill changes, and dynamic loop removal in generated code. Only a worthwhile
result should proceed to a separate, same-condition serving test and Draft PR.
