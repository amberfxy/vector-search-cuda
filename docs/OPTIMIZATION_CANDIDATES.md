# Optimization candidates (not implemented yet)

Controlled list for **after** Nsight evidence arrives. Do **not** implement
all of these. Preserve legacy + current resident baseline; add optimized
kernels as separate variants.

## A. Current: thread-per-vector (baseline)

**Status:** shipped (naive + shared-query tiled).

| | |
|---|---|
| Expected benefit | Already measured; resident E2E win is architectural |
| Memory | 1 thread / vector; strided candidate reads along dim |
| Occupancy | 256 threads/block; smem = `dim*4` (tiled) |
| Applies | All dims currently used (384/768/1024) |
| Correctness risk | Low (tested) |
| Benchmark | `bench_harness --mode resident --method tiled` |

## B. Warp-per-vector

Threads in a warp cooperate on one candidate; warp shuffle reduction.

| | |
|---|---|
| Expected benefit | Better coalescing along `dim` if consecutive threads load consecutive dims |
| Memory | Contiguous loads within a vector; fewer outstanding vectors per warp |
| Tradeoffs | Fewer vectors in flight → possible occupancy / latency-hiding change; more complex indexing |
| Applies | Best when `dim` is multiple of 32 (or handle tails) |
| Correctness | Tail dims, warp divergence at partial warps |
| Benchmark | New `GpuKernelKind::WarpPerVector` vs resident tiled; same N/dim matrix |

## C. Block-per-vector

One block owns one candidate; shared-memory / block reduction over dim.

| | |
|---|---|
| Expected benefit | Coalesced dim loads; classical reduction lecture pattern |
| Memory | Shared partial sums; higher smem; fewer blocks for large N (one block per vector) |
| Tradeoffs | At N=1M, grid = 1M blocks — launch overhead / occupancy tradeoffs; may need multi-vector per block |
| Applies | Especially larger `dim` (768/1024) |
| Correctness | Non-multiple-of-block/warp `dim`; reduction associativity vs 1e-4 tol |
| Benchmark | Separate kernel variant; A/B vs resident tiled |

## D. Vectorized loads (`float4`)

| | |
|---|---|
| Expected benefit | Fewer load instructions / better transaction packing **if** aligned |
| Memory | Requires 16-byte alignment and `dim % 4 == 0` (384/768/1024 OK) |
| Tradeoffs | Alignment assumptions; careful remainder handling if dim not multiple of 4 |
| Applies | Current dims are multiples of 4 |
| Correctness | Misaligned store base → silent wrong results — must validate |
| Benchmark | Only if ncu shows load instruction / sector pressure |

## E. Improved reduction (warp/block tree)

| | |
|---|---|
| Expected benefit | Needed for B/C; alone does little if still thread-per-vector serial dim loop |
| Tradeoffs | FP non-associativity — keep tol 1e-4 |
| Benchmark | Coupled with B or C |

## F. GPU top-k / avoid full score D2H

| | |
|---|---|
| Expected benefit | Cut D2H of N floats; matter more if kernel becomes much faster |
| Priority now | **Low** — measured resident scores D2H ≪ kernel at 1M×384 |
| Revisit | After a kernel optimization makes D2H a larger E2E fraction |
| Correctness | Exact top-k vs CPU `partial_sort`; wire tests before claiming |

## Selection rule

After an Nsight report is provided:

1. Map evidence → one of H1–H5 in `KERNEL_AUDIT.md`
2. Pick **one** candidate (usually B or C if coalescing/efficiency is poor; D only if load metrics justify)
3. Implement as a **new** kernel enum / file — do not delete baseline
4. Correctness vs CPU, then Phase 5 harness CSV (never overwrite Phase 4 files)
