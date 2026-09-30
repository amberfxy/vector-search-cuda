# Phase 1 / Phase 4 benchmark output schema

Harness: `bench_harness` (see `src/benchmark/bench_harness.cpp`).

## CSV (`results/phase1_*.csv` or path from `--csv`)

One row per (mode, method, N, dim, metric) aggregate over measured iterations.

| Column | Type | Meaning |
|---|---|---|
| `mode` | string | `legacy` (re-upload corpus every query) or `resident` (`GpuVectorIndex`) |
| `method` | string | `gpu_naive` or `gpu_tiled` (kernel family used) |
| `num_vectors` | int | Corpus size N |
| `dim` | int | Embedding dimension D |
| `metric` | string | `l2` or `cosine` |
| `batch_size` | int | Query batch size (currently always `1`) |
| `concurrency` | int | Concurrent requests (currently always `1`) |
| `warmup` | int | Warm-up iterations discarded |
| `iterations` | int | Measured iterations included in stats |
| `index_build_ms` | float | One-time resident index construction (H2D corpus); `0` for legacy |
| `alloc_mean_ms` | float | Mean allocation stage (legacy); `0` when N/A |
| `corpus_h2d_mean_ms` | float | Mean corpus H2D (legacy per-query; resident build reported in `index_build_ms`) |
| `query_h2d_mean_ms` | float | Mean query H2D |
| `kernel_mean_ms` | float | Mean kernel-only (`cudaEvent`) |
| `kernel_p50_ms` | float | Kernel p50 |
| `kernel_p95_ms` | float | Kernel p95 |
| `kernel_p99_ms` | float | Kernel p99 |
| `scores_d2h_mean_ms` | float | Mean scores D2H |
| `free_mean_ms` | float | Mean cudaFree / cleanup (legacy) |
| `e2e_mean_ms` | float | Mean end-to-end per-query latency |
| `e2e_p50_ms` | float | End-to-end p50 |
| `e2e_p95_ms` | float | End-to-end p95 |
| `e2e_p99_ms` | float | End-to-end p99 |
| `e2e_min_ms` | float | End-to-end min |
| `e2e_max_ms` | float | End-to-end max |
| `qps_from_e2e_mean` | float | `1000 / e2e_mean_ms` (single-stream) |
| `gpu_mem_used_bytes` | int64 | `cudaMemGetInfo` used bytes after setup (best-effort) |

## JSON

Same fields as an array of objects under key `"runs"`, plus metadata:

```json
{
  "schema_version": 1,
  "hardware_note": "filled by harness if cudaGetDeviceProperties succeeds",
  "runs": [ { "...": "..." } ]
}
```

## Notes

- Warm-up samples are **excluded** from all percentiles and means.
- Do not hard-code results into documentation; commit CSV/JSON from real GPU runs.
- If CUDA is unavailable, the harness exits with a non-zero status and writes nothing.
