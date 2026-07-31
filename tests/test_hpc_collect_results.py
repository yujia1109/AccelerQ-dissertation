from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


HPC = Path(__file__).resolve().parents[1] / "hpc" / "original_only"
sys.path.insert(0, str(HPC))

import collect_results  # noqa: E402


MODES = ("fixed", "ml", "random", "zero")


class StrictCompleteBlockTests(unittest.TestCase):
    def collect(self, manifest_mutator=None, status_mutator=None) -> dict[str, str]:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        manifest = root / "task_manifest.tsv"
        results = root / "results"
        rows: list[dict[str, str]] = []
        for task_id, mode in enumerate(MODES):
            row = {
                "task_id": str(task_id),
                "experiment_id": "test_locked",
                "block_id": "04qubits_05__r00",
                "prefix": "04qubits_05",
                "n_qubits": "4",
                "replicate_id": "0",
                "mode": mode,
                "config_json": "[4,1,1,100,0.001,0,100,100000,1e-6,5,128,2,0]",
                "config_sha256": "config-hash",
                "generator_pool_seed": "101",
                "sampler_seed": "202",
                "init_seed": "303",
                "timeout_seconds": "1800",
                "hamiltonian_sha256": "hamiltonian-hash",
                "model_sha256": "model-hash",
                "code_commit": "abc123",
            }
            rows.append(row)
        if manifest_mutator is not None:
            manifest_mutator(rows)
        with manifest.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            task_dir = results / "tasks" / collect_results.task_directory_name(row)
            task_dir.mkdir(parents=True)
            payload = {
                "status": "completed",
                "summary": {"best_energy": -1.25, "final_energy": -1.2},
            }
            if status_mutator is not None:
                status_mutator(row, payload)
            (task_dir / "attempt_status.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        with patch.object(
            sys,
            "argv",
            [
                "collect_results.py",
                "--task-manifest",
                str(manifest),
                "--results-dir",
                str(results),
            ],
        ):
            collect_results.main()
        with (results / "paired_blocks.csv").open(newline="") as source:
            return next(csv.DictReader(source))

    def test_fully_locked_finite_block_is_strict(self) -> None:
        block = self.collect()
        self.assertEqual(block["strict_complete"], "True")

    def test_any_lock_mismatch_is_not_strict(self) -> None:
        for field in (
            "config_sha256",
            "generator_pool_seed",
            "sampler_seed",
            "init_seed",
            "hamiltonian_sha256",
            "model_sha256",
            "code_commit",
        ):
            with self.subTest(field=field):
                block = self.collect(
                    manifest_mutator=lambda rows, field=field: rows[-1].__setitem__(
                        field, "mismatch"
                    )
                )
                self.assertEqual(block["strict_complete"], "False")

    def test_non_finite_energy_is_not_strict(self) -> None:
        def set_nan(row, payload) -> None:
            if row["mode"] == "zero":
                payload["summary"]["final_energy"] = "nan"

        block = self.collect(status_mutator=set_nan)
        self.assertEqual(block["finite_energies"], "False")
        self.assertEqual(block["strict_complete"], "False")


if __name__ == "__main__":
    unittest.main()
