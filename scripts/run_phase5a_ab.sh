#!/usr/bin/env bash
# Phase 5A focused A/B: resident tiled baseline vs resident warp-per-vector.
# Requires a CUDA build of bench_harness and a real GPU (e.g. Colab T4).
# Does NOT invent numbers if CUDA is unavailable — exits non-zero.
#
# Writes a NEW CSV (does not overwrite results/phase1_benchmark.csv).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${BENCH_BIN:-./build/bench_harness}"
CSV="${CSV:-results/phase5a_ab_1m_d384.csv}"
JSON="${JSON:-results/phase5a_ab_1m_d384.json}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-50}"
N="${N:-1000000}"
DIM="${DIM:-384}"

if [[ ! -x "$BIN" ]]; then
  echo "Missing $BIN — build first:"
  echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
  exit 1
fi

mkdir -p results
# Fresh focused A/B file for this experiment (Phase 4 CSV stays untouched).
rm -f "$CSV" "$JSON"

echo "=== Phase 5A A/B: resident tiled vs warp ==="
echo "bin=$BIN N=$N dim=$DIM warmup=$WARMUP iters=$ITERS metric=l2"
echo "csv=$CSV"
echo

"$BIN" \
  --num-vectors "$N" \
  --dim "$DIM" \
  --warmup "$WARMUP" \
  --iterations "$ITERS" \
  --metric l2 \
  --mode resident \
  --method tiled,warp \
  --csv "$CSV" \
  --json "$JSON" \
  --no-append

echo
echo "Done. CSV: $CSV"
echo "JSON: $JSON"
echo "Phase 4 baseline CSV (untouched): results/phase1_benchmark.csv"
echo "Fill docs/PHASE5_WARP_PER_VECTOR.md RESULT table from this CSV — do not invent."
