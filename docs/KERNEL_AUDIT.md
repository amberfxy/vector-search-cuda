# Kernel audit (pre–Nsight optimization)

Audit of the **current resident baseline** (`GpuVectorIndex` + tiled/naive
device launches). Distinguishes **MEASURED FACT** (from code or
`results/phase1_benchmark.csv` on Tesla T4) from **HYPOTHESIS** (not yet
profiler-proven).

## Measured facts (T4, 1M × 384, L2, tiled)

From `results/phase1_benchmark.csv` (resident vs legacy):

| Path | corpus H2D | kernel mean | E2E mean | E2E p95 | QPS |
|---|---|---|---|---|---|
| Legacy | 362.128 ms | 28.6631 ms | 396.708 ms | 429.956 ms | 2.52075 |
| Resident | 0 (per query) | 20.6963 ms | 21.6077 ms | 21.8454 ms | 46.2798 |

- One-time resident index build: **360.33 ms**
- Resident query H2D ≈ **0.0113 ms**
- Resident kernel ≈ **96%** of per-query E2E (20.6963 / 21.6077)
- E2E speedup ≈ **18.36×** is **end-to-end architecture**, not a kernel speedup

## Current execution model

### 1. Kernel execution model — FACT

- **One thread → one candidate vector** (`i = blockIdx.x * blockDim.x + threadIdx.x`)
- Each thread serially loops over all `dim` dimensions and writes `out_scores[i]`
- Tiled variant: query staged once per block in `__shared__` before the dim loop
- Naive variant: every thread reads `query[d]` from global memory

### 2. Thread / block mapping — FACT

- `threads_per_block = 256`
- `blocks = ceil(N / 256)`
- Launch: `<<<blocks, 256, dim * sizeof(float)>>>` for tiled (dynamic smem)

### 3. Memory layout — FACT

- Row-major flat store: candidate `i` at `store + i * dim`
- Query: length-`dim` vector
- Scores: length-`N` float32 buffer

### 4. Global memory access pattern — FACT (pattern); efficiency — HYPOTHESIS

**FACT:** At fixed dimension `d`, consecutive threads `i, i+1, …` read
`store[(i)*dim + d]`, `store[(i+1)*dim + d]`, … — addresses are **`dim`
floats apart**.

**HYPOTHESIS (unproven):** this strided pattern reduces coalescing / sector
efficiency and limits DRAM throughput. Requires Nsight evidence.

### 5. Shared-memory use — FACT

- Tiled kernels: `extern __shared__ float shared_query[]` sized `dim * 4` bytes
- Cooperative load: `for (d = threadIdx.x; d < dim; d += blockDim.x)`
- `__syncthreads()` before distance loop
- Candidates are **not** staged in shared memory

### 6. Reduction strategy — FACT

- **Per-thread serial sum** over dimensions (no warp/block tree reduction)
- L2: `sum += diff*diff` then `sqrtf`
- Cosine: serial dot + norms, then divide (0 if denom == 0)

### 7–8. Output behavior — FACT

- Kernel writes **full N scores** to device buffer
- `GpuVectorIndex::search` **D2H copies all N floats** every call
- No top-k in the resident hot path (Thrust top-k stub exists but is unwired)

### 9. Synchronization — FACT

In `GpuVectorIndex::search` (staged timing path used by harness):

- `cudaEventSynchronize` after query H2D
- after kernel
- after scores D2H  

Legacy staged path also syncs between alloc / corpus H2D / free.

Kernel internals: `__syncthreads()` after shared-query load (tiled only).

### 10. Per-kernel temporary allocation — FACT

- **Resident search:** no `cudaMalloc` / `cudaFree` per query; reuses
  `d_store_`, `d_query_`, `d_scores_`
- **Legacy wrappers:** malloc + corpus H2D + free every call
- Kernel itself does not allocate device heap memory

### 11. Performance hypotheses (for Nsight — not claims)

| ID | Hypothesis | Why plausible | How to confirm |
|---|---|---|---|
| H1 | Kernel is memory-bound on candidate traffic | AI ≈ 0.75 FLOP/byte (analytic); resident E2E ≈ kernel | DRAM throughput vs peak; compute throughput |
| H2 | Strided candidate loads hurt efficiency | Access pattern above | Load/store efficiency, sectors/req |
| H3 | Occupancy / register pressure limit SM util | Unknown | Achieved occupancy, regs/thread |
| H4 | Score D2H is minor after residency | Measured D2H ≪ kernel at 1M×384 | Already suggested by stage CSV; confirm under nsys |
| H5 | Shared-query helps little at large N | Historical naive≈tiled kernel at 100K–1M | Compare naive vs tiled under ncu |

**Do not treat H1–H5 as facts until a report is committed.**

## Next step

Profile the **resident tiled L2** path at **1M × 384** with Nsight Compute
(see `docs/NSIGHT_PROFILING.md`). Choose **one** optimization candidate from
`docs/OPTIMIZATION_CANDIDATES.md` only after evidence exists.
