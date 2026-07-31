# Original-only HPC experiment

This directory runs the dissertation's controlled configuration- and
seed-locked Original-only sensitivity experiment on Slurm. Historical and
Hospital data are not included or merged into its denominator.

Every task writes to its own directory. The `fixed`, `ml`, `random`, and `zero`
modes in a matched block share the Hamiltonian hash, 13 algorithm parameters,
code commit, generator-pool seed, sampling seed, and random-initialisation seed.
Seeds are deterministic and independent between blocks.

## Files

- `generate_tasks.py`: freezes task/config/seed/input hashes in a TSV manifest.
- `run_task.py`: runs exactly one manifest row and records completed, timeout, or
  exception status.
- `original_array.sbatch`: generic Slurm array worker.
- `submit_original.sh`: creates/reuses the frozen manifest and submits the array.
- `collect_results.py`: writes `attempts.csv`, `paired_blocks.csv`, and
  `coverage.csv`.

## 1. Runtime preflight

The submission script first schedules a small runtime-preflight job on a compute
node. It requires scikit-learn 1.7.0 and successfully loads the frozen model;
the experiment array has an `afterok` dependency and cannot start if preflight
fails. This supports clusters such as KCL CREATE, where Singularity is only
available on compute nodes.

You can also check the container interactively on CREATE:

```bash
srun --partition=cpu --account=YOUR_ACCOUNT --time=00:10:00 --mem=2G \
  singularity exec /path/to/accelerq.sif \
  python3 -c "import sklearn; print(sklearn.__version__)"
```

The required output is `1.7.0`.

## 2. Four-task smoke test

```bash
cd /path/to/AccelerQ-dissertation/hpc/original_only

EXPERIMENT_ID=rescue_smoke_20260731_v1 \
PREFIXES=04qubits_05 \
MODES=fixed,ml,random,zero \
REPLICATES=1 \
TIMEOUT_SECONDS=1800 \
SLURM_TIME=00:40:00 \
MAX_CONCURRENT=4 \
CONTAINER_IMAGE=/path/to/accelerq.sif \
SBATCH_ARGS="--partition=YOUR_PARTITION --account=YOUR_ACCOUNT" \
./submit_original.sh
```

Collect the smoke results and require `Strict complete blocks: 1/1` before a
main submission.

## 3. Recommended 24-task main experiment

```bash
EXPERIMENT_ID=rescue_locked_main_20260731_v1 \
PREFIXES=04qubits_05,06qubits_06,14qubits_05,16qubits_05,20qubits_00,20qubits_01 \
MODES=fixed,ml,random,zero \
REPLICATES=1 \
TIMEOUT_SECONDS=21600 \
SLURM_TIME=06:30:00 \
MAX_CONCURRENT=8 \
CONTAINER_IMAGE=/path/to/accelerq.sif \
SBATCH_ARGS="--partition=YOUR_PARTITION --account=YOUR_ACCOUNT" \
./submit_original.sh
```

Use `REPLICATES=2` for the 48-task version.

## 4. Independent 24-qubit stress experiment

```bash
EXPERIMENT_ID=rescue_locked_24q_20260731_v1 \
PREFIXES=24qubits_05,24qubits_06,24qubits_08 \
MODES=fixed,ml,random,zero \
REPLICATES=1 \
TIMEOUT_SECONDS=21600 \
SLURM_TIME=06:30:00 \
MAX_CONCURRENT=3 \
CONTAINER_IMAGE=/path/to/accelerq.sif \
SBATCH_ARGS="--partition=YOUR_PARTITION --account=YOUR_ACCOUNT" \
./submit_original.sh
```

If the cluster environment already contains all dependencies, omit
`CONTAINER_IMAGE`; the same 1.7.0 preflight still applies.

If you only have a Docker archive, build the SIF once on the HPC login/build
node according to local policy:

```bash
apptainer build accelerq.sif docker-archive://accelerq-docker.tar
```

## 5. Collect results

```bash
python3 collect_results.py \
  --task-manifest ../../results/EXPERIMENT_ID/task_manifest.tsv \
  --results-dir ../../results/EXPERIMENT_ID
```

The scientific denominator is in `coverage.csv`. `paired_blocks.csv` marks a
block strict only when every required mode completed with finite best/final
energies and all frozen configuration, seed, input/model hash, and code-commit
fields agree.

## Useful overrides

```text
REPO_ROOT         repository path on the cluster
EXPERIMENT_ID     unique immutable batch name
RESULTS_DIR       output directory
MODEL_PATH        frozen original_full model pickle
PREFIXES          comma-separated Hamiltonian prefixes
MODES             fixed,ml,random,zero
REPLICATES        paired replicates per Hamiltonian; default 2
TIMEOUT_SECONDS   scientific per-attempt timeout; default 21600
SLURM_TIME        scheduler wall time; default 06:30:00
CPUS_PER_TASK     default 4
MEMORY            default 64G
MAX_CONCURRENT    array concurrency cap; default 6
CONTAINER_IMAGE   optional Apptainer/Singularity image
SBATCH_ARGS       partition/account/QoS flags
EXPECTED_SKLEARN_VERSION  required runtime version; default 1.7.0
PREFLIGHT_TIME    runtime-preflight wall time; default 00:10:00
PREFLIGHT_MEMORY  runtime-preflight memory; default 2G
```

Do not change `PREFIXES`, `MODES`, seeds, or timeouts after the task manifest is
created. To change the design, use a new `EXPERIMENT_ID`.
