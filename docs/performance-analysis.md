# Performance analysis

## Separation of concerns

| Phase | What changed | T4 evidence (historical) |
|---|---|---|
| **4 — residency** | Remove per-query corpus H2D/alloc | E2E ~396.7 → ~21.6 ms (~18.4×) |
| **5A — warp mapping** | Contiguous dim loads + shfl reduce | Kernel ~20.9 → ~5.82 ms (~3.59×) |

**Do not multiply** these into one cumulative E2E claim from mixed runs.

## Why residency helped (measured stages)

On 1M×384 L2, legacy corpus H2D dominated (~362 ms). Resident path sets
per-query corpus H2D to 0; kernel became ~96% of E2E. That is **architecture**,
not a claim that FLOPs got 18× faster.

## Why warp-per-vector helped (Nsight)

Baseline tiled (thread-per-vector) at 1M×384:

- Achieved occupancy ~98% → **not occupancy-bound**  
- Eligible warps/scheduler ~0.05; no-eligible ~97.75%  
- LG memory instruction-queue stall ~87% of inter-issue cycles  
- DRAM throughput ~44%

Warp-per-vector (measured):

- DRAM ~97%; eligible warps/scheduler ~0.67; cycles/issued insn ~350 → ~18.7  
- Occupancy **fell** slightly (~92%) while runtime improved  

Interpretation: access-pattern / memory-instruction pathology improved;
scoreboard stalls remain; DRAM near peak. See `PHASE5_WARP_PER_VECTOR.md`.

## What we do **not** claim without new profiles

- That pinned memory / streams / CUDA Graphs help on this workload  
- That float4 or block-per-vector is faster (not implemented as default)  
- Cross-GPU absolute millisecond targets  

## Profiling recipe

Documented in `docs/NSIGHT_PROFILING.md` (`profile_kernel` + `ncu` sections).
