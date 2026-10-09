# Colab GPU diagnostics / telemetry

Lightweight, **single-GPU** diagnostics for this repo. Intended for Google
Colab (Tesla T4) or any machine with `nvidia-smi` / NVML.

## What it does

1. Samples GPU util / memory / temperature / power (`pynvml` or `nvidia-smi`)
2. Sweeps batch sizes via `./build/bench_batch` while sampling telemetry
3. Runs controlled stress cases (`diagnostics_stress` + legacy vs resident)
4. Writes CSV/JSON + a short `REPORT.md` with symptom → likely cause mapping

## What it does **not** do

- Overwrite `results/phase1_benchmark.csv` or invent Phase 4/5 speedups
- Claim multi-GPU, NVLink, PCIe topology, firmware, or datacenter ops experience

## Colab quickstart

```python
# 1) Runtime → Change runtime type → GPU
!git clone https://github.com/amberfxy/vector-search-cuda.git
%cd vector-search-cuda
!sudo apt-get update -qq && sudo apt-get install -y -qq cmake

!cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$(which nvcc)
!cmake --build build -j 2 --target bench_batch bench_harness diagnostics_stress

!pip install -q pynvml  # optional; nvidia-smi fallback works without it

!python3 scripts/colab_diagnostics.py \
  --num-vectors 100000 --dim 384 \
  --batches 1,8,32,128 --method tiled \
  --warmup 3 --iterations 10
```

Or open [`notebooks/Colab_GPU_Diagnostics.ipynb`](../notebooks/Colab_GPU_Diagnostics.ipynb).

## Outputs

```
results/diagnostics/<UTC-timestamp>/
  experiments.csv
  experiments.json
  REPORT.md
  batch_benchmark_with_telemetry.csv
results/diagnostics/latest/   # copy of the latest run
```

## Failure-mode mapping (heuristic)

| Observation | Likely cause tag |
|---|---|
| `cudaMalloc` / OOM in detail | `memory_pressure_oom` |
| NaN / null / zero-dim throw | `invalid_input` |
| Legacy E2E ≫ kernel | `transfer_bottleneck` |
| E2E ≫ baseline batch=1 on same run | `kernel_or_e2e_regression` |
| mem_used / mem_total > ~92% | `memory_pressure_near_limit` |

## CLI

```bash
python3 scripts/colab_diagnostics.py --help
python3 scripts/colab_diagnostics.py --skip-stress
python3 scripts/colab_diagnostics.py --method warp --batches 1,8,32
```

## Troubleshooting (REPORT says skipped / baseline missing)

Your report means **`bench_batch` / `diagnostics_stress` were not built** — usually
cmake ran **CPU-only** because `nvcc` was missing, even though `nvidia-smi` works.

```python
!nvidia-smi
!nvcc --version          # must succeed
!rm -rf build
!cmake -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20
# MUST see: "CUDA compiler found: building GPU kernels"
# NOT:      "No CUDA compiler found -- building CPU-only"
!cmake --build build -j 2 --target bench_batch bench_harness diagnostics_stress
!ls -l build/bench_batch build/bench_harness build/diagnostics_stress
!python3 scripts/colab_diagnostics.py --skip-build --num-vectors 100000 --dim 384 --batches 1,8,32,128
```

If `nvcc` is missing on Colab GPU runtime, install a toolkit matching the image, e.g.:

```python
!apt-get install -y -qq nvidia-cuda-toolkit
# or use Colab's CUDA path:
import os
print(os.environ.get("CUDA_HOME"), os.environ.get("PATH"))
!ls /usr/local/cuda/bin/nvcc
```
