import csv
import glob
import os
from collections import defaultdict


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def summarise_by_prefix(results_dir: str, expected_runs: int) -> str:
    summary_files = sorted(glob.glob(os.path.join(results_dir, "summary_*q_*.csv")))
    grouped = defaultdict(list)

    for path in summary_files:
        filename = os.path.basename(path)
        parts = filename.replace(".csv", "").split("_")
        n_qubits = parts[1]
        init_mode = parts[2]

        with open(path, newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                run_prefix = row.get("run_prefix", "unknown")
                grouped[(run_prefix, n_qubits, init_mode)].append(row)

    output_path = os.path.join(results_dir, "prefix_summary.csv")
    fieldnames = [
        "run_prefix",
        "n_qubits",
        "init_mode",
        "successful_records",
        "expected_runs",
        "records_per_expected_run",
        "successful_run_ids",
        "mean_best_energy",
        "min_best_energy",
        "mean_final_energy",
        "mean_iterations",
        "mean_active_params",
    ]

    rows = []
    for (run_prefix, n_qubits, init_mode), data in sorted(grouped.items()):
        best_energy = [float(row["best_energy"]) for row in data]
        final_energy = [float(row["final_energy"]) for row in data]
        iterations = [int(row["iterations_completed"]) for row in data]
        active_params = [int(row["active_param_count"]) for row in data]
        run_ids = sorted({row.get("run_id", "unknown") for row in data}, key=lambda value: int(value) if value.isdigit() else -1)
        successful_records = len(data)

        rows.append({
            "run_prefix": run_prefix,
            "n_qubits": n_qubits,
            "init_mode": init_mode,
            "successful_records": successful_records,
            "expected_runs": expected_runs,
            "records_per_expected_run": successful_records / expected_runs if expected_runs else float("nan"),
            "successful_run_ids": ";".join(run_ids),
            "mean_best_energy": mean(best_energy),
            "min_best_energy": min(best_energy),
            "mean_final_energy": mean(final_energy),
            "mean_iterations": mean(iterations),
            "mean_active_params": mean(active_params),
        })

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    for row in rows:
        print(
            row["run_prefix"],
            row["init_mode"],
            "records=",
            row["successful_records"],
            "expected_runs=",
            row["expected_runs"],
            "best=",
            row["mean_best_energy"],
            "iters=",
            row["mean_iterations"],
            "params=",
            row["mean_active_params"],
        )
    return output_path


if __name__ == "__main__":
    default_results_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../results/holdout_test_baselines")
    )
    results_dir = os.environ.get("RESULTS_DIR", default_results_dir)
    expected_runs = int(os.environ.get("EXPECTED_RUNS", "10"))
    summarise_by_prefix(results_dir, expected_runs)
