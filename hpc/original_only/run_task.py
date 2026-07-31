#!/usr/bin/env python3
"""Run one frozen Original-only task from a TSV manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_task(path: Path, task_id: int) -> dict[str, str]:
    with path.open(newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            if int(row["task_id"]) == task_id:
                return row
    raise IndexError(f"task_id {task_id} not found in {path}")


def task_directory_name(task: dict[str, str]) -> str:
    return (
        f"{int(task['task_id']):04d}_"
        f"{task['prefix']}_r{int(task['replicate_id']):02d}_{task['mode']}"
    )


def read_summary(task_dir: Path, n_qubits: int, mode: str) -> dict[str, str]:
    path = task_dir / f"summary_{n_qubits}q_{mode}.csv"
    if not path.is_file():
        return {}
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    return rows[-1] if rows else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    task_manifest = args.task_manifest.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    model_path = args.model_path.expanduser()
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    model_path = model_path.resolve()

    task = read_task(task_manifest, args.task_id)
    task_dir = results_dir / "tasks" / task_directory_name(task)
    task_dir.mkdir(parents=True, exist_ok=True)
    status_path = task_dir / "attempt_status.json"

    if status_path.is_file():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") in {"completed", "timeout"}:
            print(f"Task {args.task_id} already {existing['status']}; skipping")
            return 0

    atomic_json(task_dir / "task.json", task)

    n_qubits = int(task["n_qubits"])
    replicate_id = int(task["replicate_id"])
    mode = task["mode"]
    prefix = task["prefix"]
    timeout_seconds = int(task["timeout_seconds"])
    generator_pool_seed = int(task["generator_pool_seed"])
    sampler_seed = int(task["sampler_seed"])
    init_seed = int(task["init_seed"])
    config = json.loads(task["config_json"])

    hamiltonian_path = repo_root / "hamiltonian" / f"{prefix}.data"
    if sha256_file(hamiltonian_path) != task["hamiltonian_sha256"]:
        raise RuntimeError(f"Hamiltonian SHA mismatch: {hamiltonian_path}")
    if mode == "ml":
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if sha256_file(model_path) != task["model_sha256"]:
            raise RuntimeError(f"Model SHA mismatch: {model_path}")

    os.environ.update(
        {
            "INIT_MODE": mode,
            "RUN_PREFIX": prefix,
            "RUN_ID": str(replicate_id),
            "INIT_RANDOM_SEED": str(init_seed),
            "INIT_FIXED_VALUE": "0.1",
            "INIT_PARAM_MODEL": str(model_path),
            "RESULTS_DIR": str(task_dir),
            "MPLCONFIGDIR": str(task_dir / ".matplotlib"),
        }
    )

    # Keep the generator-pool, sampling, and random-initialisation streams
    # independent, while sharing each frozen seed across modes in the block.
    random.seed(generator_pool_seed)

    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(repo_root))
    # Isolate legacy logger.txt and every append-only CSV to this task directory.
    os.chdir(task_dir)

    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    status: dict[str, Any] = {
        **task,
        "status": "running",
        "started_at_utc": started_at,
        "task_dir": str(task_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    }
    atomic_json(status_path, status)

    def handle_scheduler_signal(signum: int, _frame: object) -> None:
        status.update(
            {
                "status": "scheduler_terminated",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - start,
                "termination_signal": signum,
            }
        )
        atomic_json(status_path, status)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_scheduler_signal)
    signal.signal(signal.SIGINT, handle_scheduler_signal)

    try:
        import numpy as np
        import stopit

        from kcl_util import process_file
        from kcl_util_adapt_vqe import compress_avqe, wrapper_avqe

        np.set_printoptions(precision=17)
        np.random.seed(sampler_seed)
        x_vec_params = np.asarray(config, dtype=float)
        if x_vec_params.shape != (13,):
            raise ValueError(f"Expected 13 hyperparameters, got {x_vec_params.shape}")
        if int(round(x_vec_params[0])) != n_qubits:
            raise ValueError("Config n_qubits does not match task n_qubits")

        loaded_n_qubits, hamiltonian = process_file(
            str(repo_root / "hamiltonian"), f"{prefix}.data"
        )
        if int(loaded_n_qubits) != n_qubits:
            raise ValueError("Loaded Hamiltonian qubit count does not match manifest")
        terms_before = len(hamiltonian.terms)
        compress_avqe(hamiltonian)
        terms_after = len(hamiltonian.terms)
        np.save(task_dir / "hyperparameters.npy", x_vec_params)

        energy = None
        with stopit.ThreadingTimeout(timeout_seconds) as timeout_context:
            energy = wrapper_avqe(x_vec_params, n_qubits, hamiltonian)

        elapsed = time.monotonic() - start
        summary = read_summary(task_dir, n_qubits, mode)
        if timeout_context.state == timeout_context.TIMED_OUT:
            final_status = "timeout"
            energy = None
        else:
            final_status = "completed"
            if energy is not None:
                np.save(task_dir / "energy.npy", np.asarray([float(energy)]))

        status.update(
            {
                "status": final_status,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                "energy": None if energy is None else float(energy),
                "terms_before_compression": terms_before,
                "terms_after_compression": terms_after,
                "summary": summary,
            }
        )
        atomic_json(status_path, status)
        print(
            f"Task {args.task_id} {final_status}: prefix={prefix} "
            f"replicate={replicate_id} mode={mode} elapsed={elapsed:.1f}s"
        )
        return 0
    except Exception as exc:
        elapsed = time.monotonic() - start
        trace = traceback.format_exc()
        (task_dir / "exception.txt").write_text(trace, encoding="utf-8")
        status.update(
            {
                "status": "exception",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        )
        atomic_json(status_path, status)
        print(trace, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
