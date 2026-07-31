"""Analytic baseline for the one-parameter ADAPT-QSCI objective."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from scipy.optimize import minimize


@dataclass(frozen=True)
class AnalyticScalarMinimum:
    theta: float
    energy: float
    amplitude: float
    is_flat: bool


@dataclass(frozen=True)
class ScalarBFGSResult:
    theta: float
    energy: float
    residual: float
    nit: int
    nfev: int
    success: bool
    retry_used: bool


def wrap_period_pi(theta: float) -> float:
    """Map an angle to the canonical interval [-pi/2, pi/2)."""
    return float((theta + math.pi / 2.0) % math.pi - math.pi / 2.0)


def scalar_objective(theta: float, a: float, b: float, gamma: float) -> float:
    """Evaluate a cos^2(theta) + b sin^2(theta) + gamma sin(theta) cos(theta)."""
    c = math.cos(theta)
    s = math.sin(theta)
    return a * c * c + b * s * s + gamma * s * c


def scalar_gradient(theta: float, a: float, b: float, gamma: float) -> float:
    """First derivative of :func:`scalar_objective`."""
    return (b - a) * math.sin(2.0 * theta) + gamma * math.cos(2.0 * theta)


def scalar_hessian(theta: float, a: float, b: float, gamma: float) -> float:
    """Second derivative of :func:`scalar_objective`."""
    return (
        2.0 * (b - a) * math.cos(2.0 * theta)
        - 2.0 * gamma * math.sin(2.0 * theta)
    )


def analytic_scalar_minimum(
    a: float,
    b: float,
    gamma: float,
    relative_flat_tolerance: float = 1e-12,
) -> AnalyticScalarMinimum:
    """Return a canonical global minimiser for the ``+ gamma`` convention.

    The objective is

        f(theta) = a cos^2(theta) + b sin^2(theta)
                   + gamma sin(theta) cos(theta).

    Pass ``-gamma`` when coefficients were formed with the opposite
    commutator/sign convention.
    """
    amplitude = math.hypot(a - b, gamma)
    scale = max(1.0, abs(a), abs(b), abs(gamma))
    is_flat = amplitude <= relative_flat_tolerance * scale

    if is_flat:
        theta = 0.0
    else:
        phase = math.atan2(gamma, a - b)
        theta = wrap_period_pi(0.5 * (phase + math.pi))

    minimum_energy = 0.5 * (a + b - amplitude)
    return AnalyticScalarMinimum(
        theta=theta,
        energy=minimum_energy,
        amplitude=amplitude,
        is_flat=is_flat,
    )


def minimize_scalar_bfgs(
    a: float,
    b: float,
    gamma: float,
    initial_theta: float,
    *,
    gradient_tolerance: float = 1e-6,
    retry_theta: float = 0.1,
) -> ScalarBFGSResult:
    """Run the shared scalar BFGS path and retry once after failure."""

    def objective(x: Sequence[float]) -> float:
        return scalar_objective(float(x[0]), a, b, gamma)

    options = {"disp": False, "gtol": gradient_tolerance}
    result = minimize(objective, [float(initial_theta)], method="BFGS", options=options)
    retry_used = not bool(result.success)
    if retry_used:
        result = minimize(objective, [float(retry_theta)], method="BFGS", options=options)

    theta = float(result.x[0])
    return ScalarBFGSResult(
        theta=theta,
        energy=float(result.fun),
        residual=abs(scalar_gradient(theta, a, b, gamma)),
        nit=int(getattr(result, "nit", -1)),
        nfev=int(getattr(result, "nfev", -1)),
        success=bool(result.success),
        retry_used=retry_used,
    )
