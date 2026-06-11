import csv
import glob
import os


def read_summary(path: str) -> list[dict]:
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def combine_summaries(output_path: str, input_paths: list[str]) -> None:
    rows = []
    seen = set()

    for path in input_paths:
        for row in read_summary(path):
            key = (row["n_qubits"], row["init_mode"])
            if key in seen:
                continue
            seen.add(key)
            row["source_summary"] = path
            rows.append(row)

    rows.sort(key=lambda row: (int(row["n_qubits"].replace("q", "")), row["init_mode"]))

    fieldnames = [
        "n_qubits",
        "init_mode",
        "runs",
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
            row["n_qubits"],
            row["init_mode"],
            "best=",
            row["mean_best_energy"],
            "iters=",
            row["mean_iterations"],
            "params=",
            row["mean_active_params"],
        )


if __name__ == "__main__":
    default_results_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../results")
    )
    results_root = os.environ.get("RESULTS_ROOT", default_results_root)
    output_path = os.environ.get(
        "OUTPUT_PATH",
        os.path.join(results_root, "four_way_summary.csv"),
    )
    input_paths = os.environ.get("SUMMARY_FILES")

    if input_paths:
        summary_files = input_paths.split(":")
    else:
        summary_files = sorted(glob.glob(os.path.join(results_root, "init_*", "combined_summary.csv")))

    combine_summaries(output_path, summary_files)
