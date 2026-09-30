# Phase 5A — Warp-per-vector L2 experiment

## Measured (Nsight Compute, T4, resident tiled, 1M × 384 L2)

From the provided Nsight report (do not confuse with normal `bench_harness` timings):

| Observation | Value |
|---|---|
| Theoretical occupancy | 100% |
| Achieved occupancy | ≈ 98.4% |
| Registers / thread | 24 |
| Eligible warps / scheduler | ≈ 0.05 |
| No eligible | ≈ 97.75% |
| LG mem instruction-queue stall | ≈ 304 cycles (~86.9% of inter-issue cycles) |
| Compute throughput | ≈ 3.8% |
| DRAM throughput | ≈ 44% |
| L1/TEX throughput | ≈ 97–98% |

**FACT:** Kernel is **not** occupancy-limited. It is heavily **latency / LG-memory
instruction-queue stall** limited (many resident warps, few eligible).

**Normal benchmark baseline (unchanged reference):** resident tiled kernel mean
**20.6963 ms**, E2E **21.6077 ms**, QPS **46.2798**. Do not replace these with
profiler durations.

## Old memory-access mapping (thread-per-vector) — FACT

Layout: row-major `store[vec * dim + d]`, `sizeof(float)=4`.

Mapping: thread `t` → candidate `t` (global thread id).

Warp 0 (threads 0–31) at loop index `d`:

| Lane | Vector | Address (byte offset from `store`) |
|---|---|---|
| 0 | 0 | `(0*dim + d)*4` |
| 1 | 1 | `(1*dim + d)*4` |
| … | … | … |
| 31 | 31 | `(31*dim + d)*4` |

Stride between consecutive lanes at fixed `d`: **`dim` floats = `dim*4` bytes**
(1536 bytes at dim=384).

**HYPOTHESIS (not proven coalescing failure):** this strided pattern contributes
to inefficient global/L1 traffic and the observed LG queue stalls. Warp-per-vector
is the controlled test of that hypothesis.

## New memory-access mapping (warp-per-vector) — FACT (implementation)

Mapping: warp `w` → candidate `w`.

Within a warp, lane `k` loads dimensions `k, k+32, k+64, …`:

At base `d0 = 0, 32, 64, …`:

| Lane | Dimension | Candidate address offset |
|---|---|---|
| 0 | d0+0 | `(vec*dim + d0+0)*4` |
| 1 | d0+1 | `(vec*dim + d0+1)*4` |
| … | … | … |
| 31 | d0+31 | `(vec*dim + d0+31)*4` (if `< dim`) |

Stride between consecutive lanes: **1 float = 4 bytes** (contiguous within the row).

Partial L2 sums reduced with `__shfl_down_sync`; lane 0 writes `out_scores[vec]`.
No atomics, no extra global temps, no shared memory in this experiment.
`dim` not divisible by 32 is handled by the strided loop bound check.

Kernel symbol: **`l2_warp_per_vector_kernel`**  
Launch helper: `launch_l2_warp_device`  
Enum: `GpuKernelKind::Warp`  
CLI: `--method warp` (L2 only)

## Hypothesis

Changing candidate access from strided-across-vectors to contiguous-within-vector
will reduce LG-memory instruction-queue stalls and improve resident kernel /
E2E latency versus tiled baseline — **to be confirmed on T4**.

## Experiment

A/B on T4 (same harness, resident mode):

1. `--method tiled` (baseline)  
2. `--method warp` (experiment)

Focused first: **N=1M, dim=384, L2**. Then expand if improved.

## Result

*(Leave blank until real T4 `bench_harness` CSV exists. Do not claim success.)*

| Metric | resident tiled | resident warp |
|---|---|---|
| kernel mean ms | 20.6963 (Phase 4 ref) | *TBD* |
| E2E mean ms | 21.6077 | *TBD* |
| QPS | 46.2798 | *TBD* |

Re-profile warp kernel with the same Nsight sections and compare eligible-warp /
LG-stall metrics to the baseline numbers above.

## What this is not

- Not float4, not block-per-vector, not top-k, not streams  
- Not a default replacement for tiled  
- Not an 18× kernel claim (that number was E2E residency)
