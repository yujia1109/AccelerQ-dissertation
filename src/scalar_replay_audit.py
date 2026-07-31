#!/usr/bin/env python3
"""Replay fixed one-dimensional ADAPT-QSCI objectives across initial values.

The expensive ADAPT-QSCI trajectory is used only to capture scalar objectives

    E(theta) = a cos(theta)^2 + b sin(theta)^2
               + gamma cos(theta) sin(theta).

This script then evaluates zero, fixed, seeded-random, ML, and analytic-oracle
initial values on exactly the same saved objective with the production BFGS
settings. No sampling, generator selection, or ansatz update is repeated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import statistics
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from scipy.optimize import OptimizeResult, minimize

from analytic_scalar_baseline import (
    analytic_scalar_minimum,
    scalar_objective,
    wrap_period_pi,
)
from build_init_param_dataset import FEATURE_COLUMNS


MODES = ("zero", "fixed", "random", "ml", "analytic_oracle")
DEFAULT_TRAINING_PREFIXES = {
    "02qubits_05",
    "04qubits_00",
    "04qubits_01",
    "04qubits_02",
    "04qubits_03",
    "04qubits_04",
    "06qubits_05",
    "07qubits_05",
    "08qubits_05",
    "10qubits_05",
    "12qubits_00",
    "12qubits_01",
    "12qubits_02",
    "12qubits_03",
    "12qubits_04",
}


def periodic_angle_distance(left: float, right: float) -> float:
    """Return the smallest absolute distance for angles equivalent modulo pi."""
    return abs(wrap_period_pi(left - right))


def stable_seed(base_seed: int, event_id: str) -> int:
    payload = f"{base_seed}|{event_id}|scalar-replay-random".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_modes(raw: str) -> list[str]:
    modes = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise ValueError(f"Unknown replay modes: {unknown}; choose from {MODES}")
    if len(modes) != len(set(modes)):
        raise ValueError("Replay modes must not contain duplicates")
    if not modes:
        raise ValueError("At least one replay mode is required")
    return modes


def load_events(paths: Sequence[Path]) -> list[dict[str, str]]:
    required = {
        "event_id",
        "run_prefix",
        "run_id",
        "source_init_mode",
        "event_index",
        "generator",
        "source_x0",
        "a",
        "b",
        "gamma",
        *FEATURE_COLUMNS,
    }
    events: list[dict[str, str]] = []
    seen: dict[str, Path] = {}
    for path in paths:
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            missing = sorted(required - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            for line_number, row in enumerate(reader, start=2):
                event_id = row["event_id"]
                if not event_id:
                    raise ValueError(f"{path}:{line_number} has an empty event_id")
                if event_id in seen:
                    raise ValueError(
                        f"Duplicate event_id {event_id!r} in {path}; "
                        f"first seen in {seen[event_id]}"
                    )
                seen[event_id] = path
                row["capture_file"] = str(path)
                events.append(row)
    if not events:
        raise ValueError("No scalar replay events were loaded")
    return events


def validate_capture_status(paths: Sequence[Path], allow_incomplete: bool) -> None:
    """Reject partial capture tasks before phase-based stratification."""
    if allow_incomplete:
        return
    invalid: list[str] = []
    for path in paths:
        status_path = path.with_name("capture_status.json")
        if not status_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            invalid.append(f"{path} ({status.get('status', 'unknown')})")
    if invalid:
        raise ValueError(
            "Incomplete source captures cannot define a true late-iteration stratum: "
            + "; ".join(invalid)
            + ". Exclude them or pass --allow-incomplete-captures for an explicitly "
            "partial exploratory analysis."
        )


def load_ml_predictor(
    model_path: Path | None,
    allow_version_mismatch: bool = False,
) -> tuple[Callable[[dict[str, str]], float] | None, dict[str, Any]]:
    if model_path is None:
        return None, {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        with model_path.open("rb") as model_file:
            payload = pickle.load(model_file)
    mismatch_details = sorted(
        {
            (
                str(getattr(item.message, "original_sklearn_version", "unknown")),
                str(getattr(item.message, "current_sklearn_version", sklearn.__version__)),
            )
            for item in caught
            if isinstance(item.message, InconsistentVersionWarning)
        }
    )
    if mismatch_details and not allow_version_mismatch:
        versions = ", ".join(
            f"trained with {original}, running with {current}"
            for original, current in mismatch_details
        )
        raise ValueError(
            f"scikit-learn model version mismatch ({versions}). Run the replay in "
            "the frozen experiment environment, or explicitly pass "
            "--allow-model-version-mismatch for a non-primary smoke test."
        )
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Unexpected model payload in {model_path}")
    payload_columns = list(payload.get("feature_columns", []))
    if payload_columns != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature columns do not match the replay schema: "
            f"model={payload_columns}, expected={FEATURE_COLUMNS}"
        )
    model = payload["model"]

    def predict(event: dict[str, str]) -> float:
        features = np.asarray(
            [[float(event[column]) for column in FEATURE_COLUMNS]], dtype=float
        )
        return float(model.predict(features)[0])

    metadata = {
        "path": str(model_path),
        "sha256": sha256_file(model_path),
        "feature_columns": payload_columns,
        "training_metrics": payload.get("metrics", {}),
        "version_mismatches": [
            {"original": original, "current": current}
            for original, current in mismatch_details
        ],
    }
    return predict, metadata


def run_bfgs(
    objective: Callable[[np.ndarray], float],
    initial_theta: float,
    retry_theta: float,
    gradient_tolerance: float,
) -> tuple[OptimizeResult, OptimizeResult, OptimizeResult | None]:
    """Run production-equivalent BFGS, preserving both calls after a retry."""
    options = {"disp": False, "gtol": gradient_tolerance}
    primary = minimize(
        objective,
        np.asarray([initial_theta], dtype=float),
        method="BFGS",
        options=options,
    )
    retry = None
    final = primary
    if not primary.success:
        retry = minimize(
            objective,
            np.asarray([retry_theta], dtype=float),
            method="BFGS",
            options=options,
        )
        final = retry
    return final, primary, retry


def initial_theta_for_mode(
    mode: str,
    event: dict[str, str],
    analytic_theta: float,
    ml_predict: Callable[[dict[str, str]], float] | None,
    fixed_value: float,
    random_low: float,
    random_high: float,
    random_seed: int,
) -> float:
    if mode == "zero":
        return 0.0
    if mode == "fixed":
        return fixed_value
    if mode == "random":
        rng = np.random.default_rng(stable_seed(random_seed, event["event_id"]))
        return float(rng.uniform(random_low, random_high))
    if mode == "ml":
        if ml_predict is None:
            raise ValueError("ML replay requires --model-path")
        return float(ml_predict(event))
    if mode == "analytic_oracle":
        return analytic_theta
    raise ValueError(f"Unknown mode: {mode}")


def replay_event(
    event: dict[str, str],
    mode: str,
    ml_predict: Callable[[dict[str, str]], float] | None,
    fixed_value: float,
    random_low: float,
    random_high: float,
    random_seed: int,
    retry_theta: float,
    gradient_tolerance: float,
    relative_flat_tolerance: float,
    equivalence_atol: float,
    equivalence_rtol: float,
) -> dict[str, Any]:
    a = float(event["a"])
    b = float(event["b"])
    gamma = float(event["gamma"])
    analytic = analytic_scalar_minimum(
        a,
        b,
        gamma,
        relative_flat_tolerance=relative_flat_tolerance,
    )
    maximum = 0.5 * (a + b + analytic.amplitude)
    initial_theta = initial_theta_for_mode(
        mode,
        event,
        analytic.theta,
        ml_predict,
        fixed_value,
        random_low,
        random_high,
        random_seed,
    )

    def objective(theta: np.ndarray) -> float:
        return scalar_objective(float(theta[0]), a, b, gamma)

    initial_objective = objective(np.asarray([initial_theta]))
    if analytic.is_flat:
        initial_normalised_regret: float | str = ""
    else:
        raw_regret = (initial_objective - analytic.energy) / analytic.amplitude
        initial_normalised_regret = min(1.0, max(0.0, float(raw_regret)))

    final, primary, retry = run_bfgs(
        objective,
        initial_theta,
        retry_theta,
        gradient_tolerance,
    )
    final_theta = float(final.x[0])
    final_objective = float(final.fun)
    residual = abs(final_objective - analytic.energy)
    equivalence_tolerance = equivalence_atol + equivalence_rtol * analytic.amplitude
    total_nfev = int(getattr(primary, "nfev", 0)) + (
        int(getattr(retry, "nfev", 0)) if retry is not None else 0
    )
    total_nit = int(getattr(primary, "nit", 0)) + (
        int(getattr(retry, "nit", 0)) if retry is not None else 0
    )

    row: dict[str, Any] = {
        "event_id": event["event_id"],
        "capture_file": event.get("capture_file", ""),
        "run_prefix": event["run_prefix"],
        "run_id": event["run_id"],
        "source_init_mode": event["source_init_mode"],
        "event_index": int(event["event_index"]),
        "n_qubits": int(float(event["n_qubits"])),
        "iteration": int(float(event["iteration"])),
        "generator_index": int(float(event["generator_index"])),
        "generator": event["generator"],
        "mode": mode,
        "a": a,
        "b": b,
        "gamma": gamma,
        "objective_range_D": analytic.amplitude,
        "is_flat": analytic.is_flat,
        "analytic_theta": analytic.theta,
        "analytic_minimum": analytic.energy,
        "analytic_maximum": maximum,
        "x0": initial_theta,
        "initial_objective": initial_objective,
        "initial_normalised_regret": initial_normalised_regret,
        "initial_periodic_angle_error": periodic_angle_distance(
            initial_theta, analytic.theta
        ),
        "final_theta": final_theta,
        "final_objective": final_objective,
        "final_analytic_residual": residual,
        "analytic_equivalent": residual <= equivalence_tolerance,
        "equivalence_tolerance": equivalence_tolerance,
        "optimizer_success": bool(final.success),
        "optimizer_status": int(final.status),
        "optimizer_message": str(final.message),
        "optimizer_retry_used": retry is not None,
        "optimizer_primary_nfev": int(getattr(primary, "nfev", 0)),
        "optimizer_retry_nfev": int(getattr(retry, "nfev", 0)) if retry is not None else 0,
        "optimizer_total_nfev": total_nfev,
        "optimizer_primary_nit": int(getattr(primary, "nit", 0)),
        "optimizer_retry_nit": int(getattr(retry, "nit", 0)) if retry is not None else 0,
        "optimizer_total_nit": total_nit,
    }
    for column in FEATURE_COLUMNS:
        row[f"feature_{column}"] = event[column]
    return row


def median(values: Iterable[float]) -> float | str:
    materialised = list(values)
    return statistics.median(materialised) if materialised else ""


def summary_row(rows: Sequence[dict[str, Any]], mode: str) -> dict[str, Any]:
    selected = [row for row in rows if row["mode"] == mode]
    regrets = [
        float(row["initial_normalised_regret"])
        for row in selected
        if row["initial_normalised_regret"] != ""
    ]
    return {
        "mode": mode,
        "events": len(selected),
        "nonflat_events": len(regrets),
        "median_initial_normalised_regret": median(regrets),
        "median_bfgs_nfev": median(
            float(row["optimizer_total_nfev"]) for row in selected
        ),
        "retry_rate": (
            sum(bool(row["optimizer_retry_used"]) for row in selected) / len(selected)
            if selected
            else ""
        ),
        "median_final_analytic_residual": median(
            float(row["final_analytic_residual"]) for row in selected
        ),
        "analytic_equivalence_rate": (
            sum(bool(row["analytic_equivalent"]) for row in selected) / len(selected)
            if selected
            else ""
        ),
        "optimizer_success_rate": (
            sum(bool(row["optimizer_success"]) for row in selected) / len(selected)
            if selected
            else ""
        ),
    }


def annotate_strata(
    rows: list[dict[str, Any]], training_prefixes: set[str]
) -> dict[str, Any]:
    trajectory_max: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        key = (row["run_prefix"], row["run_id"], row["source_init_mode"])
        trajectory_max[key] = max(trajectory_max[key], int(row["event_index"]))

    range_by_event = {
        row["event_id"]: float(row["objective_range_D"])
        for row in rows
        if not bool(row["is_flat"])
    }
    ranges = list(range_by_event.values())
    if ranges:
        q25, q75 = (float(value) for value in np.quantile(ranges, [0.25, 0.75]))
    else:
        q25 = q75 = 0.0

    for row in rows:
        n_qubits = int(row["n_qubits"])
        if n_qubits < 12:
            row["size_band"] = "small_lt12q"
        elif n_qubits == 12:
            row["size_band"] = "main_12q"
        elif 20 <= n_qubits <= 24:
            row["size_band"] = "extrapolation_20_24q"
        else:
            row["size_band"] = "other"

        row["distribution"] = (
            "training_prefix"
            if row["run_prefix"] in training_prefixes
            else "held_out_prefix"
        )
        if row["run_prefix"] == "24qubits_06":
            row["named_case"] = "ml_positive_24qubits_06"
        elif row["run_prefix"] == "24qubits_08":
            row["named_case"] = "counterexample_24qubits_08"
        else:
            row["named_case"] = "other"

        key = (row["run_prefix"], row["run_id"], row["source_init_mode"])
        maximum_event_index = trajectory_max[key]
        if maximum_event_index <= 1:
            progress = 0.0
        else:
            progress = (int(row["event_index"]) - 1) / (maximum_event_index - 1)
        row["source_progress"] = progress
        if progress < 1.0 / 3.0:
            row["iteration_phase"] = "early"
        elif progress < 2.0 / 3.0:
            row["iteration_phase"] = "middle"
        else:
            row["iteration_phase"] = "late"

        if bool(row["is_flat"]):
            row["d_band"] = "flat"
        elif float(row["objective_range_D"]) <= q25:
            row["d_band"] = "small_D"
        elif float(row["objective_range_D"]) >= q75:
            row["d_band"] = "large_D"
        else:
            row["d_band"] = "middle_D"

    return {"objective_range_D_q25": q25, "objective_range_D_q75": q75}


def build_stratified_summary(
    rows: Sequence[dict[str, Any]], modes: Sequence[str]
) -> list[dict[str, Any]]:
    dimensions = (
        "run_prefix",
        "size_band",
        "distribution",
        "named_case",
        "iteration_phase",
        "d_band",
        "source_init_mode",
    )
    output: list[dict[str, Any]] = []
    for dimension in dimensions:
        values = sorted({str(row[dimension]) for row in rows})
        for value in values:
            subset = [row for row in rows if str(row[dimension]) == value]
            for mode in modes:
                output.append(
                    {
                        "stratum_type": dimension,
                        "stratum_value": value,
                        **summary_row(subset, mode),
                    }
                )
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", nargs="+", type=Path, help="Captured scalar-event CSV files")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--fixed-value", type=float, default=0.1)
    parser.add_argument("--random-low", type=float, default=-1.0)
    parser.add_argument("--random-high", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=20260726)
    parser.add_argument("--retry-theta", type=float, default=0.1)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-6)
    parser.add_argument("--relative-flat-tolerance", type=float, default=1e-12)
    parser.add_argument("--equivalence-atol", type=float, default=1e-10)
    parser.add_argument("--equivalence-rtol", type=float, default=1e-8)
    parser.add_argument("--allow-incomplete-captures", action="store_true")
    parser.add_argument("--allow-model-version-mismatch", action="store_true")
    parser.add_argument(
        "--training-prefixes",
        default=",".join(sorted(DEFAULT_TRAINING_PREFIXES)),
        help="Comma-separated prefixes used to train the frozen ML model",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    modes = parse_modes(args.modes)
    if args.random_low >= args.random_high:
        raise ValueError("--random-low must be smaller than --random-high")
    if args.gradient_tolerance <= 0 or args.relative_flat_tolerance < 0:
        raise ValueError("Tolerances must be positive (flat tolerance may be zero)")
    event_paths = [path.expanduser().resolve() for path in args.events]
    output_dir = args.output_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve() if args.model_path else None
    if "ml" in modes and model_path is None:
        raise ValueError("The default five-mode replay requires --model-path")

    validate_capture_status(event_paths, args.allow_incomplete_captures)
    events = load_events(event_paths)
    ml_predict, model_metadata = load_ml_predictor(
        model_path if "ml" in modes else None,
        allow_version_mismatch=args.allow_model_version_mismatch,
    )
    results: list[dict[str, Any]] = []
    for event in events:
        for mode in modes:
            results.append(
                replay_event(
                    event=event,
                    mode=mode,
                    ml_predict=ml_predict,
                    fixed_value=args.fixed_value,
                    random_low=args.random_low,
                    random_high=args.random_high,
                    random_seed=args.random_seed,
                    retry_theta=args.retry_theta,
                    gradient_tolerance=args.gradient_tolerance,
                    relative_flat_tolerance=args.relative_flat_tolerance,
                    equivalence_atol=args.equivalence_atol,
                    equivalence_rtol=args.equivalence_rtol,
                )
            )

    training_prefixes = {
        value.strip() for value in args.training_prefixes.split(",") if value.strip()
    }
    stratum_metadata = annotate_strata(results, training_prefixes)
    overall = [summary_row(results, mode) for mode in modes]
    stratified = build_stratified_summary(results, modes)

    write_csv(output_dir / "scalar_replay_event_results.csv", results)
    write_csv(output_dir / "scalar_replay_summary.csv", overall)
    write_csv(output_dir / "scalar_replay_stratified_summary.csv", stratified)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in event_paths
        ],
        "event_count": len(events),
        "result_count": len(results),
        "allow_incomplete_captures": args.allow_incomplete_captures,
        "allow_model_version_mismatch": args.allow_model_version_mismatch,
        "modes": modes,
        "bfgs": {
            "method": "BFGS",
            "gradient_tolerance": args.gradient_tolerance,
            "retry_theta": args.retry_theta,
            "nfev_definition": "primary plus retry function evaluations",
        },
        "initialisation": {
            "fixed_value": args.fixed_value,
            "random_low": args.random_low,
            "random_high": args.random_high,
            "random_seed": args.random_seed,
        },
        "analytic_equivalence": {
            "atol": args.equivalence_atol,
            "rtol_times_objective_range_D": args.equivalence_rtol,
            "relative_flat_tolerance": args.relative_flat_tolerance,
        },
        "normalised_regret_definition": "(E(x0)-E_min)/(E_max-E_min)",
        "training_prefixes": sorted(training_prefixes),
        "model": model_metadata,
        "strata": stratum_metadata,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scalar_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Replayed {len(events)} fixed objectives across {len(modes)} modes")
    print(f"Wrote results to {output_dir}")
    for row in overall:
        print(
            row["mode"],
            f"events={row['events']}",
            f"median_regret={row['median_initial_normalised_regret']}",
            f"median_nfev={row['median_bfgs_nfev']}",
            f"retry_rate={row['retry_rate']}",
            f"equivalence_rate={row['analytic_equivalence_rate']}",
        )


if __name__ == "__main__":
    main()
