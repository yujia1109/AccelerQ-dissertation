#!/usr/bin/env python3
"""Generate a frozen Original-only HPC task manifest.

Each task is exactly one (Hamiltonian, replicate, initialisation mode) attempt.
All modes in the same block share the same algorithm configuration and sampler
seed.  The random-initialisation seed is deterministic and independent between
blocks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PREFIXES = [
    # New / incomplete held-out instances.
    "04qubits_05",
    "16qubits_05",
    "20qubits_00",
    "20qubits_01",
    "24qubits_05",
    "24qubits_08",
    "28qubits_00",
    "28qubits_01",
    # Anchors already present in the strict historical analysis.
    "06qubits_06",
    "14qubits_05",
    "20qubits_05",
    "24qubits_06",
]

DEFAULT_MODES = ["fixed", "ml", "random", "zero"]
VALID_MODES = {"zero", "fixed", "ml", "random", "analytic"}

# Hamiltonians used to train original_full_20260625.  They are forbidden from
# the held-out task manifest unless --allow-training-prefixes is explicit.
ORIGINAL_TRAINING_PREFIXES = {
    "02qubits_05",
    "04qubits_00",
    "04qubits_01",
    "04qubits_02",
    "04qubits_03",
    "04qubits_04",
    "06qubits_05",
    "07qubits_05",
    "08qubits_05",
    "10qubits_05",
    "12qubits_00",
    "12qubits_01",
    "12qubits_02",
    "12qubits_03",
    "12qubits_04",
}

FIELDNAMES = [
    "task_id",
    "experiment_id",
    "block_id",
    "prefix",
    "n_qubits",
    "replicate_id",
    "mode",
    "config_json",
    "config_sha256",
    "generator_pool_seed",
    "sampler_seed",
    "init_seed",
    "timeout_seconds",
    "hamiltonian_sha256",
    "model_sha256",
    "code_commit",
]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    # NumPy legacy/global seeds must be in [0, 2**32 - 1].
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def git_value(repo_root: Path, *args: str, default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def default_config(n_qubits: int) -> list[float]:
    # Identical to generate_hyper_params_avqe for run_id 0/1.  Replicates are
    # repeated stochastic attempts, not mode-specific hyperparameter draws.
    return [
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
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument(
        "--model-path",
        default="results/original_full_report_20260625/train_init_model/init_param_model.pkl",
    )
    parser.add_argument("--allow-training-prefixes", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    hamiltonian_dir = repo_root / "hamiltonian"
    model_path = Path(args.model_path).expanduser()
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    model_path = model_path.resolve()

    prefixes = parse_csv(args.prefixes)
    modes = parse_csv(args.modes)
    if not prefixes:
        raise ValueError("At least one prefix is required")
    if len(prefixes) != len(set(prefixes)):
        raise ValueError("Duplicate prefixes are not allowed")
    if not modes or any(mode not in VALID_MODES for mode in modes):
        raise ValueError(f"Modes must be drawn from {sorted(VALID_MODES)}")
    if len(modes) != len(set(modes)):
        raise ValueError("Duplicate modes are not allowed")
    if args.replicates < 1:
        raise ValueError("replicates must be >= 1")
    if args.timeout_seconds < 20:
        raise ValueError("timeout-seconds must be >= 20")
    if not model_path.is_file() and "ml" in modes:
        raise FileNotFoundError(f"ML model not found: {model_path}")
    if output.exists() and not args.force:
        raise FileExistsError(f"Manifest already exists: {output}")

    forbidden = sorted(set(prefixes) & ORIGINAL_TRAINING_PREFIXES)
    if forbidden and not args.allow_training_prefixes:
        raise ValueError(
            "Held-out manifest contains model-training Hamiltonians: "
            + ",".join(forbidden)
        )

    prefix_metadata: dict[str, dict[str, object]] = {}
    pattern = re.compile(r"^(\d{2})qubits_(\d{2})$")
    for prefix in prefixes:
        match = pattern.fullmatch(prefix)
        if not match:
            raise ValueError(f"Invalid prefix: {prefix}")
        n_qubits = int(match.group(1))
        path = hamiltonian_dir / f"{prefix}.data"
        if not path.is_file():
            raise FileNotFoundError(path)
        prefix_metadata[prefix] = {
            "n_qubits": n_qubits,
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    code_commit = git_value(repo_root, "rev-parse", "HEAD")
    tracked_status = git_value(
        repo_root, "status", "--porcelain", "--untracked-files=no", default=""
    )
    if tracked_status:
        raise RuntimeError(
            "Refusing to freeze a manifest from a tracked dirty worktree:\n"
            + tracked_status
        )
    model_sha256 = sha256_file(model_path) if model_path.is_file() else ""
    created_at = datetime.now(timezone.utc).isoformat()

    rows: list[dict[str, object]] = []
    task_id = 0
    for prefix in prefixes:
        n_qubits = int(prefix_metadata[prefix]["n_qubits"])
        config = default_config(n_qubits)
        config_json = json.dumps(config, separators=(",", ":"))
        config_sha256 = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        for replicate_id in range(args.replicates):
            block_id = f"{prefix}__r{replicate_id:02d}"
            generator_pool_seed = stable_seed(
                args.experiment_id, prefix, replicate_id, "generator-pool"
            )
            sampler_seed = stable_seed(args.experiment_id, prefix, replicate_id, "sampler")
            init_seed = stable_seed(args.experiment_id, prefix, replicate_id, "init")
            for mode in modes:
                rows.append(
                    {
                        "task_id": task_id,
                        "experiment_id": args.experiment_id,
                        "block_id": block_id,
                        "prefix": prefix,
                        "n_qubits": n_qubits,
                        "replicate_id": replicate_id,
                        "mode": mode,
                        "config_json": config_json,
                        "config_sha256": config_sha256,
                        "generator_pool_seed": generator_pool_seed,
                        "sampler_seed": sampler_seed,
                        "init_seed": init_seed,
                        "timeout_seconds": args.timeout_seconds,
                        "hamiltonian_sha256": prefix_metadata[prefix]["sha256"],
                        "model_sha256": model_sha256,
                        "code_commit": code_commit,
                    }
                )
                task_id += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    experiment_manifest = {
        "schema_version": 2,
        "created_at_utc": created_at,
        "experiment_id": args.experiment_id,
        "repo_root_at_generation": str(repo_root),
        "code_commit": code_commit,
        "tracked_worktree_dirty": bool(tracked_status),
        "tracked_worktree_status": tracked_status.splitlines(),
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "prefixes": prefixes,
        "modes": modes,
        "replicates": args.replicates,
        "timeout_seconds": args.timeout_seconds,
        "task_count": len(rows),
        "hamiltonians": prefix_metadata,
        "training_prefix_guard": sorted(ORIGINAL_TRAINING_PREFIXES),
        "task_manifest": str(output),
    }
    manifest_json = output.with_name("experiment_manifest.json")
    manifest_json.write_text(
        json.dumps(experiment_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(rows)} tasks to {output}")
    print(f"Wrote experiment metadata to {manifest_json}")


if __name__ == "__main__":
    main()
