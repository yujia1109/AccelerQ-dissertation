#!/usr/bin/env python3
"""Collect Original-only task statuses into auditable CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def task_directory_name(task: dict[str, str]) -> str:
    return (
        f"{int(task['task_id']):04d}_"
        f"{task['prefix']}_r{int(task['replicate_id']):02d}_{task['mode']}"
    )


def read_tasks(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()

    task_manifest = args.task_manifest.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    tasks = read_tasks(task_manifest)
    required_modes = sorted({task["mode"] for task in tasks})

    attempts: list[dict[str, Any]] = []
    by_block: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    coverage: dict[str, Counter[str]] = defaultdict(Counter)

    for task in tasks:
        task_dir = results_dir / "tasks" / task_directory_name(task)
        status_path = task_dir / "attempt_status.json"
        if status_path.is_file():
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            status_name = payload.get("status", "unknown")
        else:
            payload = {}
            status_name = "missing"
        summary = payload.get("summary") or {}
        row = {
            "task_id": task["task_id"],
            "experiment_id": task["experiment_id"],
            "block_id": task["block_id"],
            "prefix": task["prefix"],
            "n_qubits": task["n_qubits"],
            "replicate_id": task["replicate_id"],
            "mode": task["mode"],
            "status": status_name,
            "elapsed_seconds": payload.get("elapsed_seconds", ""),
            "energy": payload.get("energy", ""),
            "best_energy": summary.get("best_energy", ""),
            "final_energy": summary.get("final_energy", ""),
            "iterations_completed": summary.get("iterations_completed", ""),
            "active_param_count": summary.get("active_param_count", ""),
            "termination_reason": summary.get("termination_reason", ""),
            "terms_before_compression": payload.get("terms_before_compression", ""),
            "terms_after_compression": payload.get("terms_after_compression", ""),
            "config_sha256": task["config_sha256"],
            "generator_pool_seed": task["generator_pool_seed"],
            "sampler_seed": task["sampler_seed"],
            "init_seed": task["init_seed"],
            "hamiltonian_sha256": task["hamiltonian_sha256"],
            "model_sha256": task["model_sha256"],
            "code_commit": task["code_commit"],
            "task_dir": str(task_dir),
            "exception_type": payload.get("exception_type", ""),
            "exception_message": payload.get("exception_message", ""),
        }
        attempts.append(row)
        by_block[(task["prefix"], task["replicate_id"])].append(row)
        coverage[task["mode"]][status_name] += 1

    attempt_fields = list(attempts[0]) if attempts else []
    write_csv(results_dir / "attempts.csv", attempts, attempt_fields)

    blocks: list[dict[str, Any]] = []
    for (prefix, replicate_id), rows in sorted(by_block.items()):
        mode_rows = {row["mode"]: row for row in rows}
        present_modes = sorted(mode_rows)
        configs = {row["config_sha256"] for row in rows}
        generator_pool_seeds = {row["generator_pool_seed"] for row in rows}
        sampler_seeds = {row["sampler_seed"] for row in rows}
        init_seeds = {row["init_seed"] for row in rows}
        hamiltonians = {row["hamiltonian_sha256"] for row in rows}
        models = {row["model_sha256"] for row in rows}
        code_commits = {row["code_commit"] for row in rows}
        completed_modes = sorted(
            mode for mode, row in mode_rows.items() if row["status"] == "completed"
        )
        finite_energies = all(
            is_finite_number(mode_rows[mode][field])
            for mode in required_modes
            if mode in mode_rows
            for field in ("best_energy", "final_energy")
        ) and present_modes == required_modes
        shared_config = len(configs) == 1 and bool(next(iter(configs), ""))
        shared_generator_pool_seed = len(generator_pool_seeds) == 1
        shared_sampler_seed = len(sampler_seeds) == 1
        shared_init_seed = len(init_seeds) == 1
        shared_hamiltonian = len(hamiltonians) == 1 and bool(
            next(iter(hamiltonians), "")
        )
        shared_model = len(models) == 1 and bool(next(iter(models), ""))
        shared_code_commit = (
            len(code_commits) == 1
            and bool(next(iter(code_commits), ""))
            and next(iter(code_commits)) != "unknown"
        )
        block: dict[str, Any] = {
            "block_id": rows[0]["block_id"],
            "prefix": prefix,
            "n_qubits": rows[0]["n_qubits"],
            "replicate_id": replicate_id,
            "required_modes": ";".join(required_modes),
            "present_modes": ";".join(present_modes),
            "completed_modes": ";".join(completed_modes),
            "shared_config": shared_config,
            "shared_generator_pool_seed": shared_generator_pool_seed,
            "shared_sampler_seed": shared_sampler_seed,
            "shared_init_seed": shared_init_seed,
            "shared_hamiltonian": shared_hamiltonian,
            "shared_model": shared_model,
            "shared_code_commit": shared_code_commit,
            "finite_energies": finite_energies,
            "strict_complete": all(
                (
                    present_modes == required_modes,
                    completed_modes == required_modes,
                    shared_config,
                    shared_generator_pool_seed,
                    shared_sampler_seed,
                    shared_init_seed,
                    shared_hamiltonian,
                    shared_model,
                    shared_code_commit,
                    finite_energies,
                )
            ),
        }
        for mode in required_modes:
            row = mode_rows.get(mode, {})
            block[f"{mode}_status"] = row.get("status", "missing")
            block[f"{mode}_best_energy"] = row.get("best_energy", "")
            block[f"{mode}_final_energy"] = row.get("final_energy", "")
            block[f"{mode}_iterations"] = row.get("iterations_completed", "")
            block[f"{mode}_active_params"] = row.get("active_param_count", "")
            block[f"{mode}_termination_reason"] = row.get(
                "termination_reason", ""
            )
        blocks.append(block)

    block_fields = list(blocks[0]) if blocks else []
    write_csv(results_dir / "paired_blocks.csv", blocks, block_fields)

    coverage_rows: list[dict[str, Any]] = []
    status_order = [
        "completed",
        "timeout",
        "exception",
        "scheduler_terminated",
        "running",
        "missing",
        "unknown",
    ]
    for mode in required_modes:
        counts = coverage[mode]
        planned = sum(counts.values())
        row: dict[str, Any] = {"mode": mode, "planned": planned}
        for status_name in status_order:
            row[status_name] = counts.get(status_name, 0)
        row["completion_rate"] = counts.get("completed", 0) / planned if planned else 0.0
        coverage_rows.append(row)
    coverage_fields = ["mode", "planned", *status_order, "completion_rate"]
    write_csv(results_dir / "coverage.csv", coverage_rows, coverage_fields)

    strict_blocks = sum(bool(row["strict_complete"]) for row in blocks)
    print(f"Wrote {results_dir / 'attempts.csv'}")
    print(f"Wrote {results_dir / 'paired_blocks.csv'}")
    print(f"Wrote {results_dir / 'coverage.csv'}")
    print(f"Strict complete blocks: {strict_blocks}/{len(blocks)}")


if __name__ == "__main__":
    main()
