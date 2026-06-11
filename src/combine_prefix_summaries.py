import csv
import os


def read_rows(path: str) -> list[dict]:
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def combine_prefix_summaries(output_path: str, input_paths: list[str]) -> str:
    rows = []
    seen = set()

    for path in input_paths:
        for row in read_rows(path):
            key = (row["run_prefix"], row["n_qubits"], row["init_mode"])
            if key in seen:
                continue
            seen.add(key)
            row["source_summary"] = path
            rows.append(row)

    rows.sort(key=lambda row: (
        row["run_prefix"],
        int(row["n_qubits"].replace("q", "")),
        row["init_mode"],
    ))

    fieldnames = [
        "run_prefix",
        "n_qubits",
        "init_mode",
        "successful_runs",
        "expected_runs",
        "success_rate",
        "successful_run_ids",
        "mean_best_energy",
        "min_best_energy",
        "mean_final_energy",
        "mean_iterations",
        "mean_active_params",
        "source_summary",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    for row in rows:
        print(
            row["run_prefix"],
            row["init_mode"],
            "success=",
            f"{row['successful_runs']}/{row['expected_runs']}",
            "best=",
            row["mean_best_energy"],
            "iters=",
            row["mean_iterations"],
            "params=",
            row["mean_active_params"],
        )
    return output_path


if __name__ == "__main__":
    input_paths_env = os.environ.get("PREFIX_SUMMARY_FILES")
    if not input_paths_env:
        raise ValueError("Set PREFIX_SUMMARY_FILES to colon-separated prefix_summary.csv files.")

    output_path = os.environ.get("OUTPUT_PATH", "../results/holdout_prefix_four_way.csv")
    input_paths = [path for path in input_paths_env.split(":") if path]
    combine_prefix_summaries(output_path, input_paths)
