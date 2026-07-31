#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

EXPERIMENT_ID="${EXPERIMENT_ID:-original_hpc_clean_v1}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/$EXPERIMENT_ID}"
TASK_MANIFEST="${TASK_MANIFEST:-$RESULTS_DIR/task_manifest.tsv}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/results/original_full_report_20260625/train_init_model/init_param_model.pkl}"
PREFIXES="${PREFIXES:-04qubits_05,16qubits_05,20qubits_00,20qubits_01,24qubits_05,24qubits_08,28qubits_00,28qubits_01,06qubits_06,14qubits_05,20qubits_05,24qubits_06}"
MODES="${MODES:-fixed,ml,random,zero}"
REPLICATES="${REPLICATES:-2}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"
MAX_CONCURRENT="${MAX_CONCURRENT:-6}"
SLURM_TIME="${SLURM_TIME:-06:30:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-64G}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
SBATCH_ARGS="${SBATCH_ARGS:-}"
EXPECTED_SKLEARN_VERSION="${EXPECTED_SKLEARN_VERSION:-1.7.0}"
PREFLIGHT_TIME="${PREFLIGHT_TIME:-00:10:00}"
PREFLIGHT_MEMORY="${PREFLIGHT_MEMORY:-2G}"

mkdir -p "$RESULTS_DIR/slurm"

if [[ ! -f "$TASK_MANIFEST" ]]; then
  python3 "$SCRIPT_DIR/generate_tasks.py" \
    --repo-root "$REPO_ROOT" \
    --output "$TASK_MANIFEST" \
    --experiment-id "$EXPERIMENT_ID" \
    --prefixes "$PREFIXES" \
    --modes "$MODES" \
    --replicates "$REPLICATES" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --model-path "$MODEL_PATH"
else
  echo "Reusing frozen task manifest: $TASK_MANIFEST"
fi

TASK_COUNT="$(($(wc -l < "$TASK_MANIFEST") - 1))"
if (( TASK_COUNT < 1 )); then
  echo "Task manifest contains no tasks: $TASK_MANIFEST" >&2
  exit 1
fi

extra_sbatch_args=()
if [[ -n "$SBATCH_ARGS" ]]; then
  # Intended for simple scheduler flags such as:
  #   --partition=cpu --account=my_account
  read -r -a extra_sbatch_args <<< "$SBATCH_ARGS"
fi

echo "Submitting $TASK_COUNT tasks"
echo "Experiment: $EXPERIMENT_ID"
echo "Results:    $RESULTS_DIR"
echo "Manifest:   $TASK_MANIFEST"

preflight_job_id="$(sbatch --parsable \
  --job-name="acc-preflight" \
  --time="$PREFLIGHT_TIME" \
  --cpus-per-task=1 \
  --mem="$PREFLIGHT_MEMORY" \
  --output="$RESULTS_DIR/slurm/preflight_%j.out" \
  --error="$RESULTS_DIR/slurm/preflight_%j.err" \
  "${extra_sbatch_args[@]}" \
  "$SCRIPT_DIR/runtime_preflight.sbatch" \
  "$REPO_ROOT" "$MODEL_PATH" "$CONTAINER_IMAGE" \
  "$EXPECTED_SKLEARN_VERSION")"
preflight_job_id="${preflight_job_id%%;*}"

array_job_id="$(sbatch --parsable \
  --job-name="acc-orig" \
  --dependency="afterok:$preflight_job_id" \
  --array="0-$((TASK_COUNT - 1))%$MAX_CONCURRENT" \
  --time="$SLURM_TIME" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEMORY" \
  --output="$RESULTS_DIR/slurm/%A_%a.out" \
  --error="$RESULTS_DIR/slurm/%A_%a.err" \
  "${extra_sbatch_args[@]}" \
  "$SCRIPT_DIR/original_array.sbatch" \
  "$REPO_ROOT" "$TASK_MANIFEST" "$RESULTS_DIR" "$MODEL_PATH" \
  "$CONTAINER_IMAGE")"
array_job_id="${array_job_id%%;*}"

echo "Runtime preflight job: $preflight_job_id"
echo "Experiment array job:  $array_job_id (afterok:$preflight_job_id)"

echo
echo "After the array finishes, collect results with:"
echo "python3 '$SCRIPT_DIR/collect_results.py' --task-manifest '$TASK_MANIFEST' --results-dir '$RESULTS_DIR'"
