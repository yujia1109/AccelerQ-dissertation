import csv
import glob
import math
import os


FEATURE_COLUMNS = [
    "n_qubits",
    "iteration",
    "generator_index",
    "energy_before",
    "abs_energy_before",
    "term_count",
    "x_count",
    "y_count",
    "z_count",
    "min_qubit",
    "max_qubit",
    "qubit_span",
    "index_sum",
    "index_mean",
    "even_index_count",
    "odd_index_count",
]


def normalise_theta(theta: float) -> float:
    """Map equivalent Pauli-rotation angles into [-pi/2, pi/2)."""
    return ((theta + math.pi / 2) % math.pi) - math.pi / 2


def parse_generator(generator: str) -> dict:
    counts = {"X": 0, "Y": 0, "Z": 0}
    indices = []

    for token in generator.split():
        if len(token) < 2:
            continue
        pauli = token[0]
        try:
            index = int(token[1:])
        except ValueError:
            continue
        if pauli in counts:
            counts[pauli] += 1
        indices.append(index)

    if indices:
        min_qubit = min(indices)
        max_qubit = max(indices)
        index_sum = sum(indices)
        index_mean = index_sum / len(indices)
        even_index_count = sum(1 for index in indices if index % 2 == 0)
        odd_index_count = len(indices) - even_index_count
    else:
        min_qubit = -1
        max_qubit = -1
        index_sum = 0
        index_mean = 0.0
        even_index_count = 0
        odd_index_count = 0

    return {
        "term_count": len(indices),
        "x_count": counts["X"],
        "y_count": counts["Y"],
        "z_count": counts["Z"],
        "min_qubit": min_qubit,
        "max_qubit": max_qubit,
        "qubit_span": max_qubit - min_qubit if indices else 0,
        "index_sum": index_sum,
        "index_mean": index_mean,
        "even_index_count": even_index_count,
        "odd_index_count": odd_index_count,
    }


def build_dataset(results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    input_files = sorted(glob.glob(os.path.join(results_dir, "init_log_*q_*.csv")))
    output_path = os.path.join(results_dir, "init_param_dataset.csv")
    rows = []

    for path in input_files:
        with open(path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for raw in reader:
                if raw["optimizer_success"] != "True":
                    continue

                generator_features = parse_generator(raw["generator"])
                theta_opt = float(raw["theta_opt"])
                row = {
                    "source_file": os.path.basename(path),
                    "source_init_mode": raw["init_mode"],
                    "x0": float(raw["x0"]),
                    "theta_opt": theta_opt,
                    "theta_target": normalise_theta(theta_opt),
                    "n_qubits": int(raw["n_qubits"]),
                    "iteration": int(raw["iteration"]),
                    "generator_index": int(raw["generator_index"]),
                    "generator": raw["generator"],
                    "energy_before": float(raw["energy_before"]),
                    "abs_energy_before": abs(float(raw["energy_before"])),
                }
                row.update(generator_features)
                rows.append(row)

    fieldnames = [
        "source_file",
        "source_init_mode",
        "x0",
        "theta_opt",
        "theta_target",
        "generator",
        *FEATURE_COLUMNS,
    ]

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print("Feature columns:")
    for column in FEATURE_COLUMNS:
        print(f"  - {column}")
    return output_path


if __name__ == "__main__":
    default_results_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../results/init_baseline")
    )
    results_dir = os.environ.get("RESULTS_DIR", default_results_dir)
    build_dataset(results_dir)
