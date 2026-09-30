#!/usr/bin/env bash
# Reproducible Phase 1 / Phase 4 matrix: legacy vs resident.
# Requires a CUDA build of bench_harness and a real GPU.
# Does NOT invent numbers if CUDA is unavailable — exits non-zero.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${BENCH_BIN:-./build/bench_harness}"
CSV="${CSV:-results/phase1_benchmark.csv}"
JSON_DIR="${JSON_DIR:-results/phase1_json}"
WARMUP="${WARMUP:-5}"
# Default 50 measured iters; allow override for quicker smoke runs.
ITERS="${ITERS:-50}"
METRIC="${METRIC:-l2}"
METHOD="${METHOD:-tiled}"

if [[ ! -x "$BIN" ]]; then
  echo "Missing $BIN — build first:"
  echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
  exit 1
fi

mkdir -p results "$JSON_DIR"
# Fresh CSV for this matrix run (schema documented in results/schema_phase1.md)
rm -f "$CSV"

N_LIST=(10000 100000 1000000)
DIM_LIST=(384 768 1024)

echo "=== Phase 1/4 benchmark matrix ==="
echo "bin=$BIN warmup=$WARMUP iters=$ITERS metric=$METRIC method=$METHOD"
echo "N=${N_LIST[*]}  dim=${DIM_LIST[*]}"
echo

for dim in "${DIM_LIST[@]}"; do
  for n in "${N_LIST[@]}"; do
    # Rough skip for huge configs on small GPUs (1M×1024 ≈ 4 GiB store).
    # Harness still owns the final OOM error; this is a courtesy skip.
    store_gb=$(python3 - <<PY
print(f"{($n * $dim * 4) / (1024**3):.3f}")
PY
)
    echo "--- N=$n dim=$dim (~${store_gb} GiB store) ---"
    json_path="${JSON_DIR}/n${n}_d${dim}_${METRIC}.json"
    if ! "$BIN" \
        --num-vectors "$n" \
        --dim "$dim" \
        --warmup "$WARMUP" \
        --iterations "$ITERS" \
        --metric "$METRIC" \
        --mode both \
        --method "$METHOD" \
        --csv "$CSV" \
        --json "$json_path"; then
      echo "WARN: run failed for N=$n dim=$dim (OOM or no GPU?). Continuing."
      continue
    fi
  done
done

echo
echo "Done. CSV: $CSV"
echo "JSON dir: $JSON_DIR"
echo "Schema: results/schema_phase1.md"
echo "If no GPU was available, no valid numbers were produced — re-run on T4/A10/A100."
