# Benchmark methodology

## Principles

1. **Never invent numbers** — if no GPU, exit non-zero.  
2. **Do not overwrite** historical Phase 4/5 CSVs when running new suites.  
3. **Do not hard-code speedups** as CI pass/fail across different GPUs.  
4. Record **GPU model + CUDA version** with every result file.

## Tools

| Binary / script | Purpose |
|---|---|
| `bench_harness` | Legacy vs resident; tiled/naive/warp; stage timings |
| `bench_batch` | Resident batched Q∈{1,8,32,128,…} |
| `scripts/run_regression_suite.sh` | Machine-local regression CSV/JSON |
| `scripts/run_phase5a_ab.sh` | Focused 1M×384 tiled vs warp |
| `scripts/faiss_baseline.py` | FAISS CPU IndexFlatL2 reference line |
| `pytorch_ext/benchmark_torch.py` | Extension vs `torch.cdist` |

## Required fields (regression / batch CSV)

- dataset size `N`, `dim`, `batch` (Q), method, metric  
- warmup / iterations  
- kernel mean + p50/p95/p99  
- E2E mean + p50/p95/p99  
- QPS (query throughput; for batch = batch × 1/e2e_mean)  
- device memory bytes (corpus + working buffers)  
- device name (and CUDA toolkit in JSON `hardware_note` when available)

## Recommended commands (GPU machine)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/test_correctness_matrix
./scripts/run_regression_suite.sh
INCLUDE_1M=1 ./scripts/run_regression_suite.sh   # if VRAM allows

./build/bench_batch --num-vectors 100000 --dim 384 --batch 32 --method warp
```

## Historical references (do not delete)

- Phase 4: `results/phase1_benchmark.csv`  
- Phase 5A narrative: `docs/PHASE5_WARP_PER_VECTOR.md`  
- New runs: `results/regression_suite.csv`, `results/batch_benchmark.csv`
