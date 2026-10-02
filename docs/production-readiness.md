# Production readiness audit — CUDA Vector Similarity Search

Date: 2026-10-02 (code audit). Historical T4 metrics below are **as previously
measured**; this document does not re-claim them without re-running.

## Scope of “production quality” for this repo

In scope: correctness, stable C++/Python API, memory safety, error handling,
batched workloads, PyTorch integration, reproducible builds, regression
benchmarks, documentation.

Out of scope (intentionally): microservices, Kubernetes, Redis, cloud
serving, distributed vector DB features.

## Verified historical measurements (do not overwrite)

| Experiment | Result (Tesla T4) |
|---|---|
| Phase 4 residency (1M×384 L2) | E2E ~396.7 → ~21.6 ms (~18.4×); QPS ~2.5 → ~46.3 |
| Phase 5A warp (same class A/B) | Kernel ~20.9 → ~5.82 ms (~3.59×); E2E ~21.9 → ~6.87 ms (~3.19×) |
| Nsight | DRAM ~44% → ~97%; occupancy not the win (~98% → ~92%) |

Sources: `results/phase1_benchmark.csv`, `docs/PHASE5_WARP_PER_VECTOR.md`.

## Audit findings

| Area | Status | Notes |
|---|---|---|
| Correctness | **Good → Stronger** | Existing GPU vs CPU tests; expanded matrix + invalid inputs |
| Memory ownership | **Good** | RAII `GpuVectorIndex`; corpus resident; buffers grown for batch |
| CUDA error handling | **Improved** | `cuda_error.hpp` with OOM/device messages + launch checks |
| Device selection | **Added** | `build(..., device=)` / `stats().device` |
| Dimension handling | **Good** | Non-multiples of 32 covered (warp loop); tested 383/385 |
| Batch handling | **Added** | `search_batch` [Q,N] scores; bench_batch |
| Memory capacity | **Improved** | Pre-flight `cudaMemGetInfo` on build; clean OOM |
| API design | **Improved** | `build` / `search` / `search_batch` / `reset` / `stats`; no `add()` |
| PyTorch integration | **Improved** | tiled + warp + batch APIs; device-resident tensors |
| Build reproducibility | **Good** | CMake CPU/CUDA; document arches 70–86 |
| Benchmark quality | **Good** | Stage timings + percentiles; regression script (machine-local) |
| Test coverage | **Improved** | `test_correctness_matrix` |
| Perf regression risk | **Mitigated** | Suite records CSV/JSON; no cross-GPU hard-coded gates |

## Gaps / limitations (honest)

- Not a vector database (no IVF/HNSW, no persistence format beyond raw floats).
- `add()` unsupported — corpus immutable after `build()`; rebuild to change data.
- Warp kernel is **L2-only**.
- Batched path = one kernel launch per query (shared H2D/D2H); not a fused multi-query GEMM.
- Pinned memory / streams / pools **not** enabled by default (no new Nsight justifying them yet).
- Top-k on GPU remains optional/unwired on the resident hot path (full score vector returned).
- CI GPU tests require a CUDA runner; CPU tests always run.

## Recommended engineer workflow

1. Build Release with CMake  
2. `ctest` / `./test_correctness_matrix` on GPU  
3. `./scripts/run_regression_suite.sh` on target GPU  
4. Compare CSV to prior runs on the **same** GPU model — never expect identical ms across SKUs  
