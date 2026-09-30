# Phase 5A GPU run status

**Hardware:** Google Colab **NVIDIA Tesla T4**  
**Workload:** resident, N=1,000,000, dim=384, L2, warmup=5, iterations=50  
**Methods:** `gpu_tiled` vs `gpu_warp` (same harness session)

## Measured summary (documented from T4 run)

| | tiled | warp |
|---|---|---|
| kernel mean ms | 20.898 | 5.823 |
| E2E mean ms | 21.881 | 6.867 |
| QPS | 45.70 | 145.63 |

Full tables + Nsight comparison: [`docs/PHASE5_WARP_PER_VECTOR.md`](../docs/PHASE5_WARP_PER_VECTOR.md).

## Artifacts still needed from Colab

The following files were **not** present in the agent workspace when docs were
updated. Copy them from the Colab run (do not invent rows):

```text
results/phase5a_ab_1m_d384.csv
results/phase5a_ab_1m_d384.json
```

Optional Nsight exports (if retained):

```text
results/ncu_resident_warp_1m_d384_l2.ncu-rep
results/ncu_resident_warp_1m_d384_l2.txt   # or print-summary log
```

**Do not overwrite** `results/phase1_benchmark.csv` (Phase 4 source of truth).
