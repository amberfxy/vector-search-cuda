#!/usr/bin/env bash
# Reproducible performance regression / batch sweep.
# Does NOT invent numbers: requires CUDA build + GPU.
# Writes machine-readable CSV; does not overwrite Phase 4/5 historical CSVs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${BENCH_BIN:-./build/bench_harness}"
OUT_CSV="${OUT_CSV:-results/regression_suite.csv}"
OUT_JSON_DIR="${OUT_JSON_DIR:-results/regression_json}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-30}"
METRIC="${METRIC:-l2}"

if [[ ! -x "$BIN" ]]; then
  echo "Missing $BIN — build first:"
  echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
  exit 1
fi

mkdir -p results "$OUT_JSON_DIR"
rm -f "$OUT_CSV"

N_LIST=(10000 100000)
DIM_LIST=(384)
# Optional 1M — set INCLUDE_1M=1
if [[ "${INCLUDE_1M:-0}" == "1" ]]; then
  N_LIST+=(1000000)
fi

echo "=== Regression suite ==="
echo "bin=$BIN warmup=$WARMUP iters=$ITERS metric=$METRIC"
echo "Note: historical Phase4/5 numbers live in phase1_benchmark.csv / PHASE5 docs."
echo "This suite records *this machine's* measurements only."
echo

# Single-query resident tiled + warp (batch=1 path via harness)
for dim in "${DIM_LIST[@]}"; do
  for n in "${N_LIST[@]}"; do
    json="${OUT_JSON_DIR}/n${n}_d${dim}_${METRIC}_tiled_warp.json"
    echo "--- N=$n dim=$dim method=tiled,warp mode=resident ---"
    "$BIN" \
      --num-vectors "$n" --dim "$dim" \
      --warmup "$WARMUP" --iterations "$ITERS" \
      --metric "$METRIC" --mode resident --method tiled,warp \
      --csv "$OUT_CSV" --json "$json" || echo "WARN: failed N=$n dim=$dim"
  done
done

# Batched search binary if present
BATCH_BIN="./build/bench_batch"
if [[ -x "$BATCH_BIN" ]]; then
  BATCH_CSV="${BATCH_CSV:-results/batch_benchmark.csv}"
  rm -f "$BATCH_CSV"
  for batch in 1 8 32 128; do
    echo "--- batch=$batch N=100000 dim=384 ---"
    "$BATCH_BIN" --num-vectors 100000 --dim 384 --batch "$batch" \
      --warmup "$WARMUP" --iterations "$ITERS" \
      --method tiled --csv "$BATCH_CSV" || true
    "$BATCH_BIN" --num-vectors 100000 --dim 384 --batch "$batch" \
      --warmup "$WARMUP" --iterations "$ITERS" \
      --method warp --csv "$BATCH_CSV" || true
  done
  echo "Batch CSV: $BATCH_CSV"
fi

echo
echo "Wrote: $OUT_CSV"
echo "Record GPU model / CUDA version in the JSON hardware_note field."
echo "Do not hard-code expected speedups across GPUs."
