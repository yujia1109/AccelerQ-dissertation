from __future__ import annotations

import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from analytic_scalar_baseline import (  # noqa: E402
    analytic_scalar_minimum,
    minimize_scalar_bfgs,
    scalar_gradient,
    scalar_hessian,
    scalar_objective,
)


class AnalyticScalarMinimumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(20260726)

    def random_coefficients(self) -> tuple[float, float, float]:
        return (
            self.rng.uniform(-10.0, 10.0),
            self.rng.uniform(-10.0, 10.0),
            self.rng.uniform(-10.0, 10.0),
        )

    def test_objective_is_pi_periodic(self) -> None:
        for _ in range(200):
            a, b, gamma = self.random_coefficients()
            theta = self.rng.uniform(-20.0, 20.0)
            self.assertAlmostEqual(
                scalar_objective(theta + math.pi, a, b, gamma),
                scalar_objective(theta, a, b, gamma),
                delta=2e-13 * max(1.0, abs(a), abs(b), abs(gamma)),
            )

    def test_analytic_solution_is_no_worse_than_dense_grid(self) -> None:
        grid_size = 16_384
        grid = [-math.pi / 2.0 + math.pi * i / grid_size for i in range(grid_size)]
        for _ in range(80):
            a, b, gamma = self.random_coefficients()
            solution = analytic_scalar_minimum(a, b, gamma)
            grid_energy = min(scalar_objective(theta, a, b, gamma) for theta in grid)
            self.assertLessEqual(
                scalar_objective(solution.theta, a, b, gamma),
                grid_energy + 1e-11,
            )

    def test_non_flat_solution_has_zero_first_derivative(self) -> None:
        for _ in range(200):
            a, b, gamma = self.random_coefficients()
            solution = analytic_scalar_minimum(a, b, gamma)
            self.assertFalse(solution.is_flat)
            self.assertLess(
                abs(scalar_gradient(solution.theta, a, b, gamma)),
                2e-13 * max(1.0, solution.amplitude),
            )

    def test_non_flat_solution_has_positive_second_derivative(self) -> None:
        for _ in range(200):
            a, b, gamma = self.random_coefficients()
            solution = analytic_scalar_minimum(a, b, gamma)
            self.assertFalse(solution.is_flat)
            self.assertGreater(scalar_hessian(solution.theta, a, b, gamma), 0.0)

    def test_minimum_energy_matches_closed_form(self) -> None:
        for _ in range(200):
            a, b, gamma = self.random_coefficients()
            solution = analytic_scalar_minimum(a, b, gamma)
            expected = 0.5 * (a + b - math.sqrt((a - b) ** 2 + gamma**2))
            self.assertAlmostEqual(solution.energy, expected, delta=2e-14)
            self.assertAlmostEqual(
                scalar_objective(solution.theta, a, b, gamma),
                expected,
                delta=2e-13 * max(1.0, abs(expected)),
            )

    def test_flat_case_has_constant_energy(self) -> None:
        a = b = -3.25
        gamma = 0.0
        solution = analytic_scalar_minimum(a, b, gamma)
        self.assertTrue(solution.is_flat)
        self.assertEqual(solution.theta, 0.0)
        self.assertEqual(solution.energy, a)
        for i in range(401):
            theta = -4.0 * math.pi + 8.0 * math.pi * i / 400.0
            self.assertAlmostEqual(scalar_objective(theta, a, b, gamma), a, delta=1e-14)


class ScalarBFGSTests(unittest.TestCase):
    def test_multiple_initial_values_report_convergence_diagnostics(self) -> None:
        a, b, gamma = 2.3, -0.7, 1.1
        analytic = analytic_scalar_minimum(a, b, gamma)
        starts = {
            "zero": 0.0,
            "fixed_0.1": 0.1,
            "random": -0.73,
            "ml": 0.82,
            "analytic": analytic.theta,
        }

        for name, initial_theta in starts.items():
            with self.subTest(name=name):
                result = minimize_scalar_bfgs(a, b, gamma, initial_theta)
                self.assertLess(result.residual, 1e-6)
                self.assertGreaterEqual(result.nit, 0)
                self.assertGreaterEqual(result.nfev, 1)
                self.assertTrue(result.success)
                self.assertFalse(result.retry_used)
                self.assertAlmostEqual(result.energy, analytic.energy, delta=2e-12)

        analytic_result = minimize_scalar_bfgs(a, b, gamma, analytic.theta)
        self.assertEqual(analytic_result.nit, 0)

    def test_failed_first_attempt_sets_retry_flag_and_returns_retry_metrics(self) -> None:
        failed = SimpleNamespace(
            success=False,
            x=[0.0],
            fun=2.3,
            nit=7,
            nfev=20,
        )
        recovered = SimpleNamespace(
            success=True,
            x=[-1.3950739297931207],
            fun=-0.797558360773,
            nit=3,
            nfev=8,
        )
        with patch("analytic_scalar_baseline.minimize", side_effect=[failed, recovered]) as mocked:
            result = minimize_scalar_bfgs(2.3, -0.7, 1.1, 0.0)

        self.assertEqual(mocked.call_count, 2)
        self.assertTrue(result.retry_used)
        self.assertTrue(result.success)
        self.assertEqual(result.nit, recovered.nit)
        self.assertEqual(result.nfev, recovered.nfev)
        self.assertLess(result.residual, 1e-12)


if __name__ == "__main__":
    unittest.main()
