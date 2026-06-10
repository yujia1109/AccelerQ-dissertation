import csv
import glob
import os
from collections import defaultdict


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def summarise(results_dir):
    summary_files = sorted(glob.glob(os.path.join(results_dir, "summary_*q_*.csv")))
    rows = []

    for path in summary_files:
        filename = os.path.basename(path)
        parts = filename.replace(".csv", "").split("_")
        n_qubits = parts[1]
        init_mode = parts[2]

        with open(path, newline="") as csv_file:
            data = list(csv.DictReader(csv_file))

        if not data:
            continue

        best_energy = [float(row["best_energy"]) for row in data]
        final_energy = [float(row["final_energy"]) for row in data]
        iterations = [int(row["iterations_completed"]) for row in data]
        active_params = [int(row["active_param_count"]) for row in data]

        rows.append({
            "n_qubits": n_qubits,
            "init_mode": init_mode,
            "runs": len(data),
            "mean_best_energy": mean(best_energy),
            "min_best_energy": min(best_energy),
            "mean_final_energy": mean(final_energy),
            "mean_iterations": mean(iterations),
            "mean_active_params": mean(active_params),
        })

    output_path = os.path.join(results_dir, "combined_summary.csv")
    fieldnames = [
        "n_qubits",
        "init_mode",
        "runs",
        "mean_best_energy",
        "min_best_energy",
        "mean_final_energy",
        "mean_iterations",
        "mean_active_params",
    ]

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote summary to {output_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    results_dir = os.environ.get("RESULTS_DIR", "../results/init_baseline")
    summarise(results_dir)