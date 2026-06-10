#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-../results/init_modes_short_$(date +%Y%m%d_%H%M%S)}"
INIT_FIXED_VALUE="${INIT_FIXED_VALUE:-0.1}"
INIT_RANDOM_SEED="${INIT_RANDOM_SEED:-1}"

TMP_DIR="$(mktemp -d)"
cp "$REPO_ROOT/src/kcl_QCELS_stage_1.py" "$TMP_DIR/kcl_QCELS_stage_1.py"
cp "$REPO_ROOT/src/kcl_adapt_vqe_stage_1.py" "$TMP_DIR/kcl_adapt_vqe_stage_1.py"

cleanup() {
  cp "$TMP_DIR/kcl_QCELS_stage_1.py" "$REPO_ROOT/src/kcl_QCELS_stage_1.py"
  cp "$TMP_DIR/kcl_adapt_vqe_stage_1.py" "$REPO_ROOT/src/kcl_adapt_vqe_stage_1.py"
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

for mode in zero fixed random; do
  echo "============================================================"
  echo "Running INIT_MODE=$mode"
  echo "Writing results to $RESULTS_DIR"
  echo "============================================================"
  INIT_MODE="$mode" \
  INIT_FIXED_VALUE="$INIT_FIXED_VALUE" \
  INIT_RANDOM_SEED="$INIT_RANDOM_SEED" \
  RESULTS_DIR="$RESULTS_DIR" \
  bash phase_1_short.sh
done

cd "$REPO_ROOT/src"
RESULTS_DIR="$RESULTS_DIR" python3 summarise_init_results.py

echo "============================================================"
echo "Completed initial-parameter short experiment."
echo "Summary:"
echo "$RESULTS_DIR/combined_summary.csv"
echo "============================================================"
