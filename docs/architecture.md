# Architecture

## What this is

A **reusable GPU distance / exact search component**: row-major float32 corpus,
L2 or cosine scores against one or many queries, with a device-resident index
API and optional PyTorch bindings.

It is **not** a distributed vector database or cloud service.

## Data layout

- Corpus: `N × dim` float32, row-major (`store[i * dim + d]`).
- Single query: `dim` floats.
- Batch queries: `Q × dim` row-major.
- Scores: single `[N]`; batch `[Q × N]` row-major.

## Execution paths

```
Legacy (bench / teaching)
  every query: alloc + corpus H2D + kernel + D2H + free

Resident (GpuVectorIndex) — production path
  build once: alloc + corpus H2D
  search / search_batch: query H2D → kernel(s) → scores D2H
  reset: free device memory
```

## Kernels (`GpuKernelKind`)

| Kind | Mapping | Notes |
|---|---|---|
| Naive | 1 thread / vector | Baseline |
| Tiled | 1 thread / vector + shared query | Default |
| Warp | 1 warp / vector, contig dim loads | L2 only; Phase 5A |

## Library surface

```cpp
auto idx = GpuVectorIndex::build(h_store, N, dim, /*device=*/-1);
idx.search(h_query, Metric::L2, h_scores, GpuKernelKind::Warp);
idx.search_batch(h_queries, Q, Metric::L2, h_scores_qn, GpuKernelKind::Tiled);
auto st = idx.stats();
idx.reset();
```

Users do not manage `cudaMalloc` for corpus/query/score buffers.

## PyTorch path

`pytorch_ext/` operates on **already CUDA** tensors (no corpus H2D in the op).
See `docs/pytorch-integration.md`.

## Related docs

- `docs/production-readiness.md` — audit  
- `docs/PHASE5_WARP_PER_VECTOR.md` — warp Nsight story  
- `docs/benchmark-methodology.md` — how to measure  
- `docs/performance-analysis.md` — measured vs hypothesis  
