#!/usr/bin/env python3
"""Run a small set of source trajectories and capture scalar replay events."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import stopit


DEFAULT_PREFIXES = (
    "06qubits_05",
    "12qubits_00",
    "20qubits_05",
    "24qubits_06",
    "24qubits_08",
)
VALID_SOURCE_MODES = {"zero", "fixed", "random", "ml", "analytic"}


def parse_csv(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo_root: Path, *args: str, default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def default_config(n_qubits: int) -> np.ndarray:
    """Return the frozen 13-value configuration used in the main comparison."""
    return np.asarray(
        [
            float(n_qubits),
            1.0,
            1.0,
            100.0,
            0.001,
            0.0,
            100.0,
            100000.0,
            1e-6,
            5.0,
            128.0,
            2.0,
            0.0,
        ],
        dtype=float,
    )


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", default="scalar_replay_capture_v1")
    parser.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES))
    parser.add_argument(
        "--source-modes",
        default="zero",
        help="Use zero for the five-run pilot; zero,ml adds a source-mode sensitivity",
    )
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("results/original_full_report_20260625/train_init_model/init_param_model.pkl"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_path = args.model_path.expanduser()
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    model_path = model_path.resolve()
    prefixes = parse_csv(args.prefixes)
    source_modes = parse_csv(args.source_modes)

    if not prefixes:
        raise ValueError("At least one prefix is required")
    if len(prefixes) != len(set(prefixes)):
        raise ValueError("Prefixes must not contain duplicates")
    if not source_modes or set(source_modes) - VALID_SOURCE_MODES:
        raise ValueError(f"Source modes must be drawn from {sorted(VALID_SOURCE_MODES)}")
    if len(source_modes) != len(set(source_modes)):
        raise ValueError("Source modes must not contain duplicates")
    if args.replicates < 1:
        raise ValueError("--replicates must be at least one")
    if args.timeout_seconds < 20:
        raise ValueError("--timeout-seconds must be at least 20")
    if "ml" in source_modes and not model_path.is_file():
        raise FileNotFoundError(model_path)

    hamiltonian_dir = repo_root / "hamiltonian"
    prefix_metadata: dict[str, Any] = {}
    for prefix in prefixes:
        hamiltonian_path = hamiltonian_dir / f"{prefix}.data"
        if not hamiltonian_path.is_file():
            raise FileNotFoundError(hamiltonian_path)
        try:
            n_qubits = int(prefix.split("qubits_", maxsplit=1)[0])
        except ValueError as exc:
            raise ValueError(f"Cannot infer qubit count from prefix {prefix!r}") from exc
        prefix_metadata[prefix] = {
            "n_qubits": n_qubits,
            "hamiltonian_path": str(hamiltonian_path),
            "hamiltonian_sha256": sha256_file(hamiltonian_path),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(repo_root))
    from kcl_util import process_file
    from kcl_util_adapt_vqe import compress_avqe, wrapper_avqe

    tasks: list[dict[str, Any]] = []
    for prefix in prefixes:
        n_qubits = int(prefix_metadata[prefix]["n_qubits"])
        for replicate_id in range(args.replicates):
            for source_mode in source_modes:
                task_id = f"{prefix}__r{replicate_id:02d}__{source_mode}"
                tasks.append(
                    {
                        "task_id": task_id,
                        "prefix": prefix,
                        "n_qubits": n_qubits,
                        "replicate_id": replicate_id,
                        "source_init_mode": source_mode,
                        "sampler_seed": stable_seed(
                            args.experiment_id, prefix, replicate_id, "sampler"
                        ),
                        "init_seed": stable_seed(
                            args.experiment_id, prefix, replicate_id, "init"
                        ),
                        "config": default_config(n_qubits).tolist(),
                    }
                )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "repo_root": str(repo_root),
        "code_commit": git_value(repo_root, "rev-parse", "HEAD"),
        "tracked_worktree_status": git_value(
            repo_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
            default="",
        ).splitlines(),
        "prefixes": prefixes,
        "source_modes": source_modes,
        "replicates": args.replicates,
        "timeout_seconds": args.timeout_seconds,
        "task_count": len(tasks),
        "prefix_metadata": prefix_metadata,
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path) if model_path.is_file() else "",
        },
        "tasks": tasks,
    }
    manifest_path = output_dir / "capture_manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"{manifest_path} already exists; use a new output directory or --force"
        )
    write_json(manifest_path, manifest)

    completed = 0
    for task in tasks:
        task_dir = output_dir / "tasks" / task["task_id"]
        task_dir.mkdir(parents=True, exist_ok=True)
        status_path = task_dir / "capture_status.json"
        capture_file = task_dir / "scalar_replay_events.csv"
        if args.force:
            generated_paths = [
                capture_file,
                status_path,
                task_dir / "capture_exception.txt",
            ]
            generated_paths.extend(task_dir.glob("init_log_*q_*.csv"))
            generated_paths.extend(task_dir.glob("summary_*q_*.csv"))
            for generated_path in generated_paths:
                if generated_path.is_file():
                    generated_path.unlink()
        if status_path.is_file() and not args.force:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "completed":
                print(f"Skipping completed capture {task['task_id']}")
                completed += 1
                continue
            if capture_file.is_file():
                raise RuntimeError(
                    f"Incomplete capture already exists for {task['task_id']}; "
                    "use a new output directory or explicitly pass --force"
                )

        os.environ.update(
            {
                "INIT_MODE": str(task["source_init_mode"]),
                "RUN_PREFIX": str(task["prefix"]),
                "RUN_ID": str(task["replicate_id"]),
                "INIT_RANDOM_SEED": str(task["init_seed"]),
                "INIT_FIXED_VALUE": "0.1",
                "INIT_PARAM_MODEL": str(model_path),
                "RESULTS_DIR": str(task_dir),
                "SCALAR_REPLAY_CAPTURE_FILE": str(capture_file),
                "MPLCONFIGDIR": str(task_dir / ".matplotlib"),
            }
        )
        random.seed(int(task["sampler_seed"]))
        np.random.seed(int(task["sampler_seed"]))

        started = datetime.now(timezone.utc).isoformat()
        start = time.monotonic()
        status: dict[str, Any] = {
            **task,
            "status": "running",
            "started_at_utc": started,
            "task_dir": str(task_dir),
            "capture_file": str(capture_file),
        }
        write_json(status_path, status)
        previous_cwd = Path.cwd()
        try:
            os.chdir(task_dir)
            loaded_n_qubits, hamiltonian = process_file(
                str(hamiltonian_dir), f"{task['prefix']}.data"
            )
            if int(loaded_n_qubits) != int(task["n_qubits"]):
                raise ValueError("Loaded Hamiltonian qubit count does not match prefix")
            terms_before = len(hamiltonian.terms)
            compress_avqe(hamiltonian)
            terms_after = len(hamiltonian.terms)
            config = np.asarray(task["config"], dtype=float)
            energy = None
            with stopit.ThreadingTimeout(args.timeout_seconds) as timeout_context:
                energy = wrapper_avqe(config, int(task["n_qubits"]), hamiltonian)
            elapsed = time.monotonic() - start
            if timeout_context.state == timeout_context.TIMED_OUT:
                final_status = "timeout"
                energy = None
            else:
                final_status = "completed"
                completed += 1
            status.update(
                {
                    "status": final_status,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": elapsed,
                    "energy": None if energy is None else float(energy),
                    "terms_before_compression": terms_before,
                    "terms_after_compression": terms_after,
                }
            )
            write_json(status_path, status)
            print(
                f"Capture {task['task_id']} {final_status}; "
                f"events={sum(1 for _ in capture_file.open()) - 1 if capture_file.is_file() else 0}; "
                f"elapsed={elapsed:.1f}s"
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            trace = traceback.format_exc()
            (task_dir / "capture_exception.txt").write_text(trace, encoding="utf-8")
            status.update(
                {
                    "status": "exception",
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": elapsed,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )
            write_json(status_path, status)
            print(trace, file=sys.stderr)
        finally:
            os.chdir(previous_cwd)

    print(f"Completed {completed}/{len(tasks)} source captures")
    return 0 if completed == len(tasks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
