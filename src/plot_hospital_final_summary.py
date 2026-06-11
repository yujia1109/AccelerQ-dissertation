import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_CSV = os.environ.get("SUMMARY_CSV", "../results/hospital_final_summary.csv")
OUT_DIR = os.environ.get("FIGURE_DIR", "../results/figures")

MODES = ["fixed", "ml", "random"]
COLORS = {
    "fixed": "#4C78A8",
    "ml": "#F58518",
    "random": "#54A24B",
}


def q_sort_key(label):
    return int(label.replace("q", ""))


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "n_qubits": row["n_qubits"],
                "init_mode": row["init_mode"],
                "mean_best_energy": float(row["mean_best_energy"]),
                "mean_iterations": float(row["mean_iterations"]),
                "mean_active_params": float(row["mean_active_params"]),
            })
    return rows


def plot_metric(rows, metric, ylabel, title, filename):
    os.makedirs(OUT_DIR, exist_ok=True)

    qubits = sorted({row["n_qubits"] for row in rows}, key=q_sort_key)
    by_key = {(row["n_qubits"], row["init_mode"]): row for row in rows}

    x = list(range(len(qubits)))
    width = 0.25
    offsets = {"fixed": -width, "ml": 0.0, "random": width}

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for mode in MODES:
        values = [
            by_key.get((q, mode), {}).get(metric, float("nan"))
            for q in qubits
        ]
        ax.bar(
            [i + offsets[mode] for i in x],
            values,
            width,
            label=mode,
            color=COLORS[mode],
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(qubits)
    ax.set_xlabel("Hospital Hamiltonian size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    output_path = os.path.join(OUT_DIR, filename)
    fig.savefig(output_path, dpi=200)
    print(f"Wrote {output_path}")


def main():
    rows = load_rows(SUMMARY_CSV)

    plot_metric(
        rows,
        "mean_best_energy",
        "Mean best energy (lower is better)",
        "Energy Accuracy Across Hospital Hamiltonians",
        "hospital_mean_best_energy.png",
    )

    plot_metric(
        rows,
        "mean_active_params",
        "Mean active parameters",
        "Ansatz Size Across Hospital Hamiltonians",
        "hospital_mean_active_params.png",
    )

    plot_metric(
        rows,
        "mean_iterations",
        "Mean iterations",
        "Convergence Iterations Across Hospital Hamiltonians",
        "hospital_mean_iterations.png",
    )


if __name__ == "__main__":
    main()
