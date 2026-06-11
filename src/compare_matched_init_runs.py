import csv
import glob
import os
from collections import defaultdict


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def load_summary_rows(results_dirs: list[str]) -> list[dict]:
    rows = []
    for results_dir in results_dirs:
        for path in sorted(glob.glob(os.path.join(results_dir, "summary_*q_*.csv"))):
            with open(path, newline="") as csv_file:
                rows.extend(csv.DictReader(csv_file))
    return rows


def compare_matched_runs(results_dirs: list[str], output_path: str) -> str:
    rows = load_summary_rows(results_dirs)
    by_problem_run = defaultdict(dict)

    for row in rows:
        run_prefix = row.get("run_prefix", "unknown")
        run_id = row.get("run_id", "unknown")
        init_mode = row["init_mode"]
        key = (run_prefix, row["n_qubits"], run_id)
        by_problem_run[key][init_mode] = row

    modes = sorted({row["init_mode"] for row in rows})
    output_rows = []

    for (run_prefix, n_qubits, run_id), mode_rows in sorted(by_problem_run.items()):
        if not all(mode in mode_rows for mode in modes):
            continue

        best_energy = min(float(mode_rows[mode]["best_energy"]) for mode in modes)
        lowest_iterations = min(int(mode_rows[mode]["iterations_completed"]) for mode in modes)
        lowest_params = min(int(mode_rows[mode]["active_param_count"]) for mode in modes)
        best_modes = [
            mode for mode in modes
            if float(mode_rows[mode]["best_energy"]) == best_energy
        ]
        lowest_iteration_modes = [
            mode for mode in modes
            if int(mode_rows[mode]["iterations_completed"]) == lowest_iterations
        ]
        lowest_param_modes = [
            mode for mode in modes
            if int(mode_rows[mode]["active_param_count"]) == lowest_params
        ]

        out = {
            "run_prefix": run_prefix,
            "n_qubits": n_qubits,
            "run_id": run_id,
            "best_energy_modes": ";".join(best_modes),
            "lowest_iterations_modes": ";".join(lowest_iteration_modes),
            "lowest_params_modes": ";".join(lowest_param_modes),
        }

        for mode in modes:
            out[f"{mode}_best_energy"] = mode_rows[mode]["best_energy"]
            out[f"{mode}_final_energy"] = mode_rows[mode]["final_energy"]
            out[f"{mode}_iterations"] = mode_rows[mode]["iterations_completed"]
            out[f"{mode}_active_params"] = mode_rows[mode]["active_param_count"]

        output_rows.append(out)

    fieldnames = [
        "run_prefix",
        "n_qubits",
        "run_id",
        "best_energy_modes",
        "lowest_iterations_modes",
        "lowest_params_modes",
    ]
    for mode in modes:
        fieldnames.extend([
            f"{mode}_best_energy",
            f"{mode}_final_energy",
            f"{mode}_iterations",
            f"{mode}_active_params",
        ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    wins = defaultdict(int)
    for row in output_rows:
        for mode in row["best_energy_modes"].split(";"):
            wins[mode] += 1

    print(f"Wrote {len(output_rows)} matched rows to {output_path}")
    print("Best-energy wins/ties on matched runs:")
    for mode in modes:
        print(f"  {mode}: {wins[mode]}")

    for mode in modes:
        energy_values = [float(row[f"{mode}_best_energy"]) for row in output_rows]
        iteration_values = [int(row[f"{mode}_iterations"]) for row in output_rows]
        param_values = [int(row[f"{mode}_active_params"]) for row in output_rows]
        print(
            f"{mode}: mean_best={mean(energy_values)}, "
            f"mean_iterations={mean(iteration_values)}, "
            f"mean_active_params={mean(param_values)}"
        )

    return output_path


if __name__ == "__main__":
    results_dirs = [
        path for path in os.environ.get(
            "RESULTS_DIRS",
            "../results/holdout_test_baselines:../results/holdout_test_ml",
        ).split(":")
        if path
    ]
    output_path = os.environ.get("OUTPUT_PATH", "../results/holdout_matched_runs.csv")
    compare_matched_runs(results_dirs, output_path)
