import os

import numpy as np

from kcl_prepare_data import miner
from kcl_util import process_file
from kcl_util_adapt_vqe import (
    compress_avqe,
    generate_hyper_params_avqe,
    wrapper_avqe,
)


def parse_csv_env(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


def run_experiment() -> None:
    np.set_printoptions(precision=17)

    prefixes = parse_csv_env("PREFIXES", "04qubits_05,06qubits_05,08qubits_05")
    modes = parse_csv_env("INIT_MODES", "zero,fixed,random")
    results_dir = os.environ.get("RESULTS_DIR", "../results/init_mode_experiment")
    hamiltonian_dir = os.environ.get("HAMILTONIAN_DIR", "../hamiltonian/")
    repeats = int(os.environ.get("REPEATS", "10"))
    timeout = int(os.environ.get("TIMEOUT_SECONDS", "660"))

    os.makedirs(results_dir, exist_ok=True)

    print(">> Initial-parameter experiment")
    print(f">> Prefixes: {prefixes}")
    print(f">> Modes: {modes}")
    print(f">> Repeats per prefix/mode: {repeats}")
    print(f">> Results dir: {results_dir}")

    for mode in modes:
        os.environ["INIT_MODE"] = mode
        for prefix in prefixes:
            os.environ["RUN_PREFIX"] = prefix
            file_name = prefix + ".data"
            print("============================================================")
            print(f">> Running prefix={prefix}, INIT_MODE={mode}")
            print("============================================================")

            n_qubits, ham = process_file(hamiltonian_dir, file_name)
            n_qubits = int(n_qubits)

            x_file = os.path.join(results_dir, f"{prefix}.{mode}.X.data")
            y_file = os.path.join(results_dir, f"{prefix}.{mode}.Y.data")
            mined = miner(
                n_qubits,
                ham,
                repeats,
                timeout,
                x_file,
                y_file,
                generate_hyper_params_avqe,
                wrapper_avqe,
                compress_avqe,
            )
            print(f">> Finished prefix={prefix}, INIT_MODE={mode}, mined={mined}")

    print(">> Done")


if __name__ == "__main__":
    run_experiment()
