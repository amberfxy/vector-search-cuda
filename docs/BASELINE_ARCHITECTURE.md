# Baseline Architecture (Phase 0 / Phase 1)

> Local engineering documentation. Describes the **pre–device-resident-index**
> architecture and how to reproduce measurements. Do not treat README Results
> tables as substitutes for re-running the harness on your GPU.

## Current execution paths

### A. Standalone legacy CUDA path (`batch_distance_*_cuda`)

```
host store (N×D) + host query (D)
  → cudaMalloc(store, query, scores)     # every query
  → H2D store (N·D·4 bytes)              # every query
  → H2D query (D·4 bytes)
  → kernel <<<ceil(N/256), 256>>>        # cudaEvent-timed
  → D2H scores (N·4 bytes)               # full score vector
  → cudaFree ×3
  → [top-k NOT in this path]
```

Used by: `bench_runner`, `bench_harness --mode legacy`, `test_correctness_gpu`.

### B. PyTorch device-resident path (`pytorch_ext/`)

```
CUDA tensors already on device
  → validate dtype/device/shape/contiguous
  → data_ptr<float>()
  → tiled kernel (no cudaMalloc / no H2D of corpus)
  → scores tensor on device
  → optional torch.topk in rerank()
```

Used by: `pytorch_ext/benchmark_torch.py`, `demo_rerank.py`.

### C. CPU baselines

- `batch_distance_singlethread` / `batch_distance_openmp` — same L2/cosine math.
- FAISS CPU `IndexFlatL2` via `scripts/faiss_baseline.py` (exact brute-force only).

## Kernel mapping (unchanged)

| Kernel | Parallelism | Query access | Candidate access |
|---|---|---|---|
| `l2_naive_kernel` / `cosine_naive_kernel` | 1 thread / candidate | global | global, stride `D` |
| `l2_tiled_kernel` / `cosine_tiled_kernel` | 1 thread / candidate | `__shared__` once/block | global, stride `D` |

Block size: 256 threads. Shared-query staging is the only shipped kernel optimization.

## Measured vs suspected bottlenecks

### Measured (from `results/benchmark.csv` on Colab Tesla T4, dim=384)

| Observation | Evidence |
|---|---|
| Wall ≫ kernel at 1M | kernel ~28 ms vs wall ~391 ms |
| Shared-query helps at small N | 10K: 0.276 → 0.178 ms kernel (~1.6×) |
| Shared-query ties at large N | 100K–1M: naive ≈ tiled kernel |
| Device-resident is faster | PyTorch ext @ 1M L2 ~21 ms (no corpus H2D) |

### Suspected (not Nsight-proven)

- Poor candidate coalescing (warp reads `D` floats apart)
- Memory-bound (AI ≈ 0.75 FLOP/byte back-of-envelope)
- Top-k not on critical path today (stub not wired)

## Why kernel-only ≠ end-to-end

Standalone wrappers **intentionally** include malloc + full corpus H2D + D2H + free
in the wall-clock region so transfer cost is visible. Kernel-only uses `cudaEvent`
around the launch only. Serving systems keep the corpus on device; the Phase 4
`GpuVectorIndex` path measures that model separately.

**Note on staged timing:** `bench_harness` inserts `cudaEventSynchronize` between
allocation / H2D / kernel / D2H / free so each stage is isolatable. That can make
staged end-to-end slightly higher than the original `bench_runner` wall number
(which syncs mainly around the kernel). Legacy vs resident A/B inside the harness
remains fair because both modes use the same staging discipline.

## Reproducing baseline numbers

```bash
# CPU + CUDA (requires nvcc + GPU)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/test_correctness_gpu          # correctness first
./build/bench_harness --help
./scripts/run_phase1_benchmark.sh     # Phase 1 matrix → results/
```

If this machine has no CUDA GPU, compile still succeeds for CPU targets;
GPU harness stages that require CUDA are skipped or fail with a clear error.
Do **not** invent numbers — re-run on T4/A10/A100 and commit CSVs from that run.

## Related schemas

See `results/schema_phase1.md` for harness output fields.
