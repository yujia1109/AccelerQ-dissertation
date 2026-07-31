from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from scalar_replay_audit import (  # noqa: E402
    periodic_angle_distance,
    replay_event,
)


def make_event(a: float, b: float, gamma: float) -> dict[str, str]:
    return {
        "event_id": "12qubits_00|0|zero|0001",
        "run_prefix": "12qubits_00",
        "run_id": "0",
        "source_init_mode": "zero",
        "event_index": "1",
        "generator": "X0 Y1",
        "source_x0": "0.0",
        "a": str(a),
        "b": str(b),
        "gamma": str(gamma),
        "n_qubits": "12",
        "iteration": "1",
        "generator_index": "3",
        "energy_before": str(a),
        "abs_energy_before": str(abs(a)),
        "term_count": "2",
        "x_count": "1",
        "y_count": "1",
        "z_count": "0",
        "min_qubit": "0",
        "max_qubit": "1",
        "qubit_span": "1",
        "index_sum": "1",
        "index_mean": "0.5",
        "even_index_count": "1",
        "odd_index_count": "1",
    }


class ScalarReplayTests(unittest.TestCase):
    def replay(self, mode: str):
        return replay_event(
            event=make_event(a=1.0, b=-1.0, gamma=0.4),
            mode=mode,
            ml_predict=(lambda _event: -0.7),
            fixed_value=0.1,
            random_low=-1.0,
            random_high=1.0,
            random_seed=7,
            retry_theta=0.1,
            gradient_tolerance=1e-6,
            relative_flat_tolerance=1e-12,
            equivalence_atol=1e-10,
            equivalence_rtol=1e-8,
        )

    def test_oracle_has_zero_initial_regret(self) -> None:
        result = self.replay("analytic_oracle")
        self.assertAlmostEqual(float(result["initial_normalised_regret"]), 0.0)
        self.assertTrue(result["analytic_equivalent"])
        self.assertEqual(result["optimizer_primary_nit"], 0)

    def test_seeded_random_is_reproducible(self) -> None:
        first = self.replay("random")
        second = self.replay("random")
        self.assertEqual(first["x0"], second["x0"])

    def test_nfev_accounting_includes_both_attempts(self) -> None:
        result = self.replay("zero")
        expected = result["optimizer_primary_nfev"] + result["optimizer_retry_nfev"]
        self.assertEqual(result["optimizer_total_nfev"], expected)
        self.assertTrue(np.isfinite(result["final_analytic_residual"]))

    def test_periodic_angle_distance(self) -> None:
        self.assertAlmostEqual(periodic_angle_distance(0.0, np.pi), 0.0)

    def test_flat_objective_omits_normalised_regret(self) -> None:
        flat = make_event(a=-2.0, b=-2.0, gamma=0.0)
        result = replay_event(
            event=flat,
            mode="zero",
            ml_predict=None,
            fixed_value=0.1,
            random_low=-1.0,
            random_high=1.0,
            random_seed=7,
            retry_theta=0.1,
            gradient_tolerance=1e-6,
            relative_flat_tolerance=1e-12,
            equivalence_atol=1e-10,
            equivalence_rtol=1e-8,
        )
        self.assertEqual(result["initial_normalised_regret"], "")
        self.assertTrue(result["analytic_equivalent"])


if __name__ == "__main__":
    unittest.main()
