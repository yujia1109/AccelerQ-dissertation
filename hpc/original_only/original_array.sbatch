#!/usr/bin/env bash
set -euo pipefail

# Scheduler resources are supplied by submit_original.sh so they can be changed
# without editing this file.

REPO_ROOT="$1"
TASK_MANIFEST="$2"
RESULTS_DIR="$3"
MODEL_PATH="$4"
CONTAINER_IMAGE="${5:-}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
RUNNER="$REPO_ROOT/hpc/original_only/run_task.py"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONHASHSEED=0

runner_args=(
  "$RUNNER"
  --repo-root "$REPO_ROOT"
  --task-manifest "$TASK_MANIFEST"
  --results-dir "$RESULTS_DIR"
  --model-path "$MODEL_PATH"
  --task-id "$TASK_ID"
)

if [[ -n "$CONTAINER_IMAGE" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    container_runtime="apptainer"
  elif command -v singularity >/dev/null 2>&1; then
    container_runtime="singularity"
  else
    echo "Neither apptainer nor singularity is available" >&2
    exit 127
  fi
  "$container_runtime" exec \
    --bind "$REPO_ROOT:$REPO_ROOT" \
    "$CONTAINER_IMAGE" \
    python3 "${runner_args[@]}"
else
  python3 "${runner_args[@]}"
fi
