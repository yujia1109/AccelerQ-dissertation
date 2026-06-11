import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

from openfermion import load_operator


def infer_n_qubits(path: Path) -> int:
    ham = load_operator(
        file_name=path.name,
        data_directory=str(path.parent),
        plain_text=False,
    )
    max_index = -1
    for term in ham.terms:
        for index, _ in term:
            max_index = max(max_index, index)
    return max_index + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy hospital Hamiltonians into AccelerQ's NNqubits_SS.data naming format."
    )
    parser.add_argument("--source", required=True, help="Folder containing hospital .data files")
    parser.add_argument("--target", default="../hamiltonian", help="AccelerQ hamiltonian folder")
    parser.add_argument("--split", required=True, choices=["train", "test"], help="Dataset split label")
    parser.add_argument("--per-size", type=int, default=5, help="How many files to copy per qubit size")
    parser.add_argument("--seed-start", type=int, default=50, help="Two-digit seed to start from")
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    selected_by_size: dict[int, list[Path]] = defaultdict(list)
    all_files = sorted(source.glob("*.data"))
    if not all_files:
        raise FileNotFoundError(f"No .data files found in {source}")

    for path in all_files:
        n_qubits = infer_n_qubits(path)
        if len(selected_by_size[n_qubits]) < args.per_size:
            selected_by_size[n_qubits].append(path)

    manifest_path = target / f"hospital_{args.split}_manifest.csv"
    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["split", "source_file", "target_file", "n_qubits", "seed"],
        )
        writer.writeheader()

        for n_qubits in sorted(selected_by_size):
            for offset, source_file in enumerate(selected_by_size[n_qubits]):
                seed = args.seed_start + offset
                if seed > 99:
                    raise ValueError("seed-start + per-size must stay within two digits")
                target_name = f"{n_qubits:02d}qubits_{seed:02d}.data"
                target_file = target / target_name
                shutil.copy2(source_file, target_file)
                writer.writerow(
                    {
                        "split": args.split,
                        "source_file": str(source_file),
                        "target_file": str(target_file),
                        "n_qubits": n_qubits,
                        "seed": seed,
                    }
                )
                print(f"{source_file.name} -> {target_name}")

    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
