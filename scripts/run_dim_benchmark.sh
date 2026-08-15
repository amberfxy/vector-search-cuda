#!/usr/bin/env bash
# Colab / CUDA-machine helper: run the full CPU/GPU/FAISS sweep at a given dim.
# Default dim=1024 matches BGE-large used by the companion Financial RAG.
#
# From repo root (after CUDA build):
#   bash scripts/run_dim_benchmark.sh
#   bash scripts/run_dim_benchmark.sh 1024 50
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results

DIM="${1:-1024}"
QUERIES="${2:-50}"
CSV="results/benchmark_dim${DIM}.csv"
CHART="results/latency_chart_dim${DIM}.png"

if [[ ! -x ./build/bench_runner ]]; then
  echo "Missing ./build/bench_runner -- build first, e.g.:"
  echo "  mkdir -p build && cd build && cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc .. && make -j"
  exit 1
fi

echo "=== bench_runner dim=${DIM} queries=${QUERIES} -> ${CSV} ==="
OMP_NUM_THREADS="$(nproc)" ./build/bench_runner "$DIM" "$QUERIES" "$CSV"

echo "=== FAISS CPU IndexFlatL2 baseline ==="
# Do not abort the script if FAISS skips/OOM on large N x dim (common on Colab).
set +e
python3 scripts/faiss_baseline.py --dim "$DIM" --queries "$QUERIES" --csv "$CSV"
faiss_rc=$?
set -e
if [[ "$faiss_rc" -ne 0 ]]; then
  echo "FAISS baseline exited with code ${faiss_rc}; continuing to plot CPU/GPU rows."
fi

echo "=== plot ==="
python3 scripts/plot_results.py --csv "$CSV" --dim "$DIM" --out "$CHART"
echo "Done: ${CSV}  ${CHART}"
