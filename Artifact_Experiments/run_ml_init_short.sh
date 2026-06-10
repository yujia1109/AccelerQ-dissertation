#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_RESULTS_DIR="${TRAIN_RESULTS_DIR:-../results/init_baseline}"
RESULTS_DIR="${RESULTS_DIR:-../results/init_ml_short_$(date +%Y%m%d_%H%M%S)}"
INIT_PARAM_MODEL="${INIT_PARAM_MODEL:-$TRAIN_RESULTS_DIR/init_param_model.pkl}"

TMP_DIR="$(mktemp -d)"
cp "$REPO_ROOT/src/kcl_QCELS_stage_1.py" "$TMP_DIR/kcl_QCELS_stage_1.py"
cp "$REPO_ROOT/src/kcl_adapt_vqe_stage_1.py" "$TMP_DIR/kcl_adapt_vqe_stage_1.py"

cleanup() {
  cp "$TMP_DIR/kcl_QCELS_stage_1.py" "$REPO_ROOT/src/kcl_QCELS_stage_1.py"
  cp "$TMP_DIR/kcl_adapt_vqe_stage_1.py" "$REPO_ROOT/src/kcl_adapt_vqe_stage_1.py"
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if [ ! -f "$INIT_PARAM_MODEL" ]; then
  echo "Could not find model at $INIT_PARAM_MODEL"
  echo "Build it first with:"
  echo "  cd $REPO_ROOT/src"
  echo "  RESULTS_DIR=$TRAIN_RESULTS_DIR python3 build_init_param_dataset.py"
  echo "  RESULTS_DIR=$TRAIN_RESULTS_DIR python3 train_init_param_model.py"
  exit 1
fi

cd "$SCRIPT_DIR"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Running INIT_MODE=ml"
echo "Using model: $INIT_PARAM_MODEL"
echo "Writing results to $RESULTS_DIR"
echo "============================================================"

INIT_MODE=ml \
INIT_PARAM_MODEL="$INIT_PARAM_MODEL" \
RESULTS_DIR="$RESULTS_DIR" \
bash phase_1_short.sh

cd "$REPO_ROOT/src"
RESULTS_DIR="$RESULTS_DIR" python3 summarise_init_results.py

echo "============================================================"
echo "Completed ML initial-parameter short experiment."
echo "Summary:"
echo "$RESULTS_DIR/combined_summary.csv"
echo "============================================================"
