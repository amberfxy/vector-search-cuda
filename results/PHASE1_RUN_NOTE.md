# Phase 1 / Phase 4 GPU run status

**Hardware:** Google Colab **NVIDIA Tesla T4**  
**Source of truth:** `results/phase1_benchmark.csv` + `results/phase1_json/`  
**Harness:** `bench_harness` / `scripts/run_phase1_benchmark.sh`  
**Config:** warmup=5, iterations=50, metric=l2, method=gpu_tiled, mode=both

Charts (generated, not hand-tuned numbers):

```bash
python3 scripts/plot_phase4_results.py
```

→ `results/phase4_resident_vs_legacy.png`  
→ `results/phase4_speedup.png`

Historical baseline CSVs/charts (`benchmark.csv`, `latency_chart.png`, etc.) are
preserved separately and were **not** overwritten by this run.
