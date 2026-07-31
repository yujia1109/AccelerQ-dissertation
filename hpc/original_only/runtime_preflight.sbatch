#!/usr/bin/env bash
set -euo pipefail

# Run this as a Slurm job because some clusters, including KCL CREATE, expose
# Singularity only on compute nodes.

REPO_ROOT="$1"
MODEL_PATH="$2"
CONTAINER_IMAGE="${3:-}"
EXPECTED_SKLEARN_VERSION="$4"

runtime_python=(python3)
if [[ -n "$CONTAINER_IMAGE" ]]; then
  if [[ ! -f "$CONTAINER_IMAGE" ]]; then
    echo "Container image not found: $CONTAINER_IMAGE" >&2
    exit 1
  fi
  if command -v apptainer >/dev/null 2>&1; then
    container_runtime="apptainer"
  elif command -v singularity >/dev/null 2>&1; then
    container_runtime="singularity"
  else
    echo "Neither apptainer nor singularity is available on the compute node" >&2
    exit 127
  fi
  runtime_python=(
    "$container_runtime" exec
    --bind "$REPO_ROOT:$REPO_ROOT"
    "$CONTAINER_IMAGE"
    python3
  )
fi

actual_sklearn_version="$(
  "${runtime_python[@]}" -c 'import sklearn; print(sklearn.__version__)'
)"
if [[ "$actual_sklearn_version" != "$EXPECTED_SKLEARN_VERSION" ]]; then
  echo "Expected scikit-learn $EXPECTED_SKLEARN_VERSION, got $actual_sklearn_version" >&2
  exit 1
fi

"${runtime_python[@]}" -c \
  'import pickle, sys; pickle.load(open(sys.argv[1], "rb")); print("Model load: OK")' \
  "$MODEL_PATH"

echo "Runtime scikit-learn: $actual_sklearn_version"
echo "Runtime preflight: PASS"
