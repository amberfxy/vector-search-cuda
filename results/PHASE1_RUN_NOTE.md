# Phase 1 / Phase 4 GPU run status

This environment built **CPU-only** (no `nvcc` / no GPU). Therefore:

- `test_correctness` and `test_bench_stats` were executed and passed.
- `bench_harness` and `test_correctness_gpu` require a CUDA build + GPU.
- **No new GPU latency numbers were invented or committed.**

## Commands to run on Colab T4 / A10 / A100

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/test_correctness_gpu
./scripts/run_phase1_benchmark.sh
# then commit results/phase1_benchmark.csv and results/phase1_json/*.json
```

After a successful GPU run, update the README "Engineering story: device-resident index"
**Result** subsection with actual measured p50/p95 and stage breakdowns from the CSV.
