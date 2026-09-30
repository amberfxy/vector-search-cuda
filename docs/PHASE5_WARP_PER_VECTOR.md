# Phase 5A — Warp-per-vector L2 experiment

## Measured (Nsight Compute, T4, resident tiled baseline, 1M × 384 L2)

From the provided Nsight report (do not confuse with normal `bench_harness` timings):

| Observation | Value |
|---|---|
| Theoretical occupancy | 100% |
| Achieved occupancy | ≈ 98.4% |
| Registers / thread | 24 |
| Eligible warps / scheduler | ≈ 0.05 |
| No eligible | ≈ 97.75% |
| LG mem instruction-queue stall | ≈ 304 cycles (~86.9% of inter-issue cycles) |
| Warp cycles per issued instruction | ≈ 350 |
| Compute throughput | ≈ 3.8% |
| DRAM throughput | ≈ 44% |
| L1/TEX throughput | ≈ 97–98% |

**FACT:** Baseline kernel is **not** occupancy-limited. It is heavily
**latency / LG-memory instruction-queue stall** limited (many resident warps,
few eligible).

**Phase 4 normal-benchmark reference (separate run; do not replace with
profiler durations):** resident tiled kernel mean **20.6963 ms**, E2E
**21.6077 ms**, QPS **46.2798**. The Phase 5A A/B below re-measured tiled in
the same harness session as warp (tiled mean **20.898 ms** in that session).

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

At each dimension step, one warp therefore issues **32 scalar float loads** whose
addresses are separated by **1536 bytes**. A single 128-byte L1 cache line (or
32-byte sector) cannot satisfy more than one of those lanes’ candidate reads at
that `d`. Over `dim=384` steps that is **32 × 384 = 12,288** candidate float
loads per warp (plus query reads from shared memory in the tiled baseline).

**HYPOTHESIS (tested by this experiment):** this strided pattern contributes
to inefficient global/L1 traffic and the observed LG queue stalls.
Warp-per-vector was the controlled test. The pre-change profiler **proved**
severe LG-memory instruction-queue stalls; it did **not** by itself prove a
coalescing verdict.

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

At each base `d0`, one warp issues **up to 32 consecutive float loads** covering
128 contiguous bytes of one candidate — a pattern that *can* map to a single
cache-line transaction when all 32 lanes are active (tail iterations when
`dim % 32 ≠ 0` activate fewer lanes). Query loads are also contiguous along dim
(from global `query[]`; this experiment intentionally does **not** stage query
in shared memory, so the A/B isolates the candidate mapping change).

Partial L2 sums reduced with `__shfl_down_sync`; lane 0 writes `out_scores[vec]`.
No atomics, no extra global temps, no shared memory in this experiment.
`dim` not divisible by 32 is handled by the strided loop bound check.

Kernel symbol: **`l2_warp_per_vector_kernel`**  
Launch helper: `launch_l2_warp_device`  
Enum: `GpuKernelKind::Warp`  
CLI: `--method warp` (L2 only; not the default)

## Hypothesis

Changing candidate access from strided-across-vectors to contiguous-within-vector
will reduce LG-memory instruction-queue stalls and improve resident kernel /
E2E latency versus tiled baseline.

## Experiment

A/B on T4 (same harness, resident mode):

1. `--method tiled` (baseline)  
2. `--method warp` (experiment)

Focused first: **N=1M, dim=384, L2**.

## Result (MEASURED — Tesla T4)

**Normal `bench_harness` A/B** (resident, N=1,000,000, dim=384, L2, warmup=5,
iterations=50). Same session for both methods — do not mix with Phase 4 CSV
numbers for speedup ratios.

| Metric | resident tiled | resident warp |
|---|---|---|
| kernel mean ms | 20.898 | **5.823** |
| kernel p50 ms | 20.760 | 5.822 |
| kernel p95 ms | 21.870 | 5.828 |
| kernel p99 ms | 22.075 | 5.834 |
| E2E mean ms | 21.881 | **6.867** |
| E2E p50 ms | 21.738 | 6.850 |
| E2E p95 ms | 22.876 | 6.951 |
| E2E p99 ms | 23.085 | 7.049 |
| QPS (from E2E mean) | 45.70 | **145.63** |

Derived (from this A/B only):

| Derived | Value |
|---|---|
| kernel speedup | 20.898 / 5.823 ≈ **3.59×** |
| E2E speedup | 21.881 / 6.867 ≈ **3.19×** |
| kernel latency reduction | ≈ **72.1%** |
| E2E latency reduction | ≈ **68.6%** |
| throughput increase | ≈ **218.6%** |

**Artifacts:** Colab CSV/JSON from this A/B were not present in the cloud agent
workspace at documentation time. Copy from Colab when available:

- `results/phase5a_ab_1m_d384.csv`
- `results/phase5a_ab_1m_d384.json`

Do **not** overwrite `results/phase1_benchmark.csv` (Phase 4).

### Nsight comparison (tiled baseline vs warp) — MEASURED

| Metric | tiled baseline | warp-per-vector |
|---|---|---|
| DRAM throughput | ≈ 44% | ≈ **97.15%** |
| Memory throughput | ≈ 49% | ≈ **97.15%** (~310.3 GB/s) |
| L1/TEX throughput | ≈ 97–98% | ≈ 57.2% |
| L1/TEX hit rate | *(not reported in baseline summary)* | ≈ 50.39% |
| Compute SM throughput | ≈ 3.8% | ≈ **56.9%** |
| One or more eligible | ≈ 2.25% | ≈ **39.26%** |
| No eligible | ≈ **97.75%** | ≈ 60.74% |
| Active warps / scheduler | ≈ 7.87 | ≈ 7.33 |
| Eligible warps / scheduler | ≈ **0.05** | ≈ **0.67** |
| Issued warp / scheduler | ≈ 0.02 | ≈ 0.39 |
| Warp cycles / issued insn | ≈ **350** | ≈ **18.68** |
| Dominant stall | LG mem insn queue ≈ **304 cycles (~86.9%)** | scoreboard dependency ≈ **12.3–12.4 cycles** |
| Theoretical occupancy | 100% | 100% |
| Achieved occupancy | ≈ 98.4% | ≈ 92.2% |
| Registers / thread | 24 | 24 |

### Interpretation supported by evidence

The experiment **strongly supports** the access-pattern hypothesis.

- Old thread-per-vector: high occupancy, severe LG-memory instruction-queue
  stalls, almost no eligible warps.
- Warp-per-vector changed candidate accesses from a **1536-byte** inter-lane
  stride (dim=384) to contiguous **4-byte** neighboring-lane accesses within
  one vector.
- DRAM utilization rose ~44% → ~97%; eligible warps/scheduler ~0.05 → ~0.67;
  no-eligible ~97.75% → ~60.74%; cycles/issued insn ~350 → ~18.7; normal
  kernel mean 20.898 → 5.823 ms.

**Do not claim occupancy improvement.** Achieved occupancy fell slightly
(~98.4% → ~92.2%) while performance improved substantially — further evidence
that occupancy was not the limiting factor.

**Do not claim all stalls were eliminated.** The warp kernel still shows
scoreboard dependency stalls, and DRAM throughput is now near device peak.

**Do not multiply Phase 4 × Phase 5** into a single cumulative E2E figure —
those are separate controlled experiments (legacy→resident architecture vs
resident tiled→warp kernel mapping).

## What this is not

- Not float4, not block-per-vector, not top-k, not streams  
- Not a silent default replacement for tiled (CLI remains `--method warp`)  
- Not an 18× kernel claim (Phase 4’s ~18× was E2E residency)
