# Nsight profiling workflow (Colab T4 / remote GPU)

Goal: profile the **resident** distance kernel at **1M × 384 L2** after the
device-resident index removed corpus H2D from the per-query path.

Related docs:

- [`KERNEL_AUDIT.md`](KERNEL_AUDIT.md) — measured facts vs hypotheses  
- [`OPTIMIZATION_CANDIDATES.md`](OPTIMIZATION_CANDIDATES.md) — do not implement until evidence exists  

Mac has no NVIDIA GPU — run these steps on Colab / a CUDA machine, then send
the report back before any kernel rewrite.

## 1. Build

```bash
cd /content/vector-search-cuda   # or your clone
git pull origin main

rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target profile_kernel test_correctness_gpu

# Optional: confirm NVTX was enabled (CMake status line)
# "NVTX ranges enabled for GpuVectorIndex / profile_kernel"
```

CPU-only machines still configure; `profile_kernel` exits with instructions
if built without CUDA.

## 2. Sanity: same resident path as harness

```bash
./build/test_correctness_gpu          # must pass before trusting profiles
./build/profile_kernel --help
./build/profile_kernel --num-vectors 10000 --dim 384 --warmup 2 --iterations 3
```

`profile_kernel` calls **`GpuVectorIndex::search`** (same API as
`bench_harness --mode resident`).

## 3. Check whether `ncu` exists

```bash
which ncu || true
ncu --version || true
ls /usr/local/cuda/bin/ncu 2>/dev/null || true
```

Colab images vary. If `ncu` is missing, install CUDA toolkit extras or use a
runtime that ships Nsight Compute CLI. If hardware performance counters are
blocked, see §7.

## 4. Recommended first profile (resident, 1M × 384, L2)

Upload corpus **once** inside the app (before warm-up). Warm-up runs before
`=== PROFILE REGION BEGIN ===`. Prefer filtering to the tiled L2 kernel:

```bash
# Section-based collection (portable across ncu versions)
ncu --target-processes all \
    --kernel-name regex:l2_tiled_kernel \
    --launch-count 20 \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --section LaunchStats \
    --section SpeedOfLight \
    -o results/ncu_resident_1m_d384_l2 \
    ./build/profile_kernel \
      --num-vectors 1000000 \
      --dim 384 \
      --metric l2 \
      --method tiled \
      --warmup 10 \
      --iterations 20
```

If sections fail on your ncu build, try `--set default` or `--set full`.

List metrics available on the installed tool:

```bash
ncu --query-metrics | head
ncu --list-sections
```

Prefer **section names** and metrics reported by *your* `ncu --version`
rather than copying deprecated metric strings from blogs.

## 5. Save / export the report

`ncu -o path` writes `path.ncu-rep` (binary). Also capture text:

```bash
ncu --import results/ncu_resident_1m_d384_l2.ncu-rep --page raw \
  > results/ncu_resident_1m_d384_l2.txt

ncu ... --print-summary per-kernel ./build/profile_kernel ... \
  | tee results/ncu_summary.txt
```

Download the `.ncu-rep` + text summary from Colab when done.

## 6. Optional: Nsight Systems timeline (NVTX)

If NVTX was enabled at build time:

```bash
nsys profile -t cuda,nvtx,osrt \
  -o results/nsys_resident_1m_d384 \
  ./build/profile_kernel --num-vectors 1000000 --dim 384 --warmup 10 --iterations 20
```

Look for ranges: `query_h2d`, `distance_kernel`, `scores_d2h`, `profile_region`.

## 7. If performance counters are unavailable

Symptoms: ncu errors about permissions, `ERR_NVGPUCTRPERM`, or empty metrics.

Try `sudo ncu ...` on machines where that is allowed (often **not** on Colab).

Fallbacks that still help:

1. Launch stats / occupancy sections only  
2. **nsys** timeline (CUDA API + NVTX) without SM counters  
3. Continue using **`bench_harness` stage timings** (already measured)  
4. Note the limitation in the report you send back — **do not invent metrics**

## 8. What to send back before kernel optimization

1. `ncu --version` output  
2. Exact `ncu` command used  
3. `*.ncu-rep` and/or exported `.txt` / `--print-summary` log  
4. Confirm workload printed `mode: resident` and `method: tiled`  
5. Any error about counters / permissions  

With that, we pick **one** candidate from `OPTIMIZATION_CANDIDATES.md`
(evidence-based), implement it as a **new** variant, and open a Phase 5
benchmark CSV — without overwriting Phase 4 results.

## 9. Phase 5A: re-profile warp-per-vector (after A/B shows improvement)

Kernel symbol: **`l2_warp_per_vector_kernel`**. Same resident path:

```bash
# Focused normal-benchmark A/B first (not profiler durations):
./scripts/run_phase5a_ab.sh
# or:
./build/bench_harness --num-vectors 1000000 --dim 384 --metric l2 \
  --mode resident --method tiled,warp --warmup 5 --iterations 50 \
  --csv results/phase5a_ab_1m_d384.csv --json results/phase5a_ab_1m_d384.json \
  --no-append

# Then Nsight on the warp kernel (same sections as baseline):
ncu --target-processes all \
    --kernel-name regex:l2_warp_per_vector_kernel \
    --launch-count 20 \
    --section SchedulerStats \
    --section WarpStateStats \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --section LaunchStats \
    --section SpeedOfLight \
    -o results/ncu_resident_warp_1m_d384_l2 \
    ./build/profile_kernel \
      --num-vectors 1000000 \
      --dim 384 \
      --metric l2 \
      --method warp \
      --warmup 10 \
      --iterations 20
```

Compare against baseline (tiled) metrics documented in
`docs/PHASE5_WARP_PER_VECTOR.md`:

| Metric | Baseline (tiled, measured) | Warp (fill after T4) |
|---|---|---|
| Eligible warps / scheduler | ≈ 0.05 | *TBD* |
| No eligible | ≈ 97.75% | *TBD* |
| LG queue stall | ≈ 304 cycles / ~86.9% | *TBD* |

Do **not** replace Phase 4 `phase1_benchmark.csv` numbers with profiler times.
