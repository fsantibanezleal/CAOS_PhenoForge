"""Interval, sharpness and calibration-error metrics for ensemble forecasts.

CRPS answers "is the whole predictive distribution good"; these answer the
questions a plant engineer actually asks next: how WIDE is the band I am being
asked to trust, does its stated coverage match reality, and what do I pay for a
miss. Reporting sharpness without calibration (or the reverse) is the classic
way to look good while being wrong, so the product ships both and the trade-off
is visible in the app.

Primary sources:
- Gneiting, T., Raftery, A.E., 2007. Strictly proper scoring rules, prediction,
  and estimation. J. Am. Stat. Assoc. 102(477), 359-378,
  DOI 10.1198/016214506000001437 (the interval score, also called the Winkler
  score, and the sharpness-subject-to-calibration paradigm).
- Winkler, R.L., 1972. A decision-theoretic approach to interval estimation.
  J. Am. Stat. Assoc. 67(337), 187-191, DOI 10.1080/01621459.1972.10481224.
- Gneiting, T., Balabdaoui, F., Raftery, A.E., 2007. Probabilistic forecasts,
  calibration and sharpness. J. R. Stat. Soc. B 69(2), 243-268,
  DOI 10.1111/j.1467-9868.2007.00587.x (PIT uniformity as the calibration
  criterion).
"""

from __future__ import annotations

import numpy as np

from phenoforge.metrics.calibration import _check, pit_values


def interval_score(ens: np.ndarray, y: np.ndarray, level: float = 0.9) -> float:
    """Mean Winkler interval score of the central `level` interval (lower is
    better; units of y).

    IS = (u - l) + (2/alpha)(l - y) 1{y < l} + (2/alpha)(y - u) 1{y > u}

    The first term rewards sharpness, the penalties price misses at the rate
    implied by the nominal level, so a method cannot win by widening its bands.
    """
    ens, y = _check(ens, y)
    if not (0.0 < level < 1.0):
        raise ValueError("level in (0, 1)")
    alpha = 1.0 - level
    lo = np.quantile(ens, alpha / 2.0, axis=0)
    hi = np.quantile(ens, 1.0 - alpha / 2.0, axis=0)
    width = hi - lo
    below = np.where(y < lo, (2.0 / alpha) * (lo - y), 0.0)
    above = np.where(y > hi, (2.0 / alpha) * (y - hi), 0.0)
    return float(np.mean(width + below + above))


def sharpness(ens: np.ndarray, level: float = 0.9) -> float:
    """Mean width of the central `level` predictive interval (units of y).

    Sharpness is a property of the FORECAST ALONE: it never looks at the
    observation, which is exactly why it must be read next to coverage.
    """
    ens = np.asarray(ens, dtype=float)
    if ens.ndim != 2 or ens.shape[0] < 2:
        raise ValueError("ens must be (n_members, n_points) with >= 2 members")
    if not (0.0 < level < 1.0):
        raise ValueError("level in (0, 1)")
    alpha = 1.0 - level
    lo = np.quantile(ens, alpha / 2.0, axis=0)
    hi = np.quantile(ens, 1.0 - alpha / 2.0, axis=0)
    return float(np.mean(hi - lo))


def pit_calibration_error(ens: np.ndarray, y: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between the PIT sample and uniform(0, 1).

    0 is perfect calibration; the maximum possible value is 1. Reported instead
    of only a histogram because a single number can be compared ACROSS methods
    and variants, which is what the benchmark matrix needs.
    """
    p = np.sort(pit_values(ens, y))
    n = p.size
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - p)
    d_minus = np.max(p - (i - 1) / n)
    return float(max(d_plus, d_minus))


def coverage_deviation(
    ens: np.ndarray,
    y: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
) -> float:
    """Mean absolute deviation of empirical coverage from nominal across levels.

    A reliability-curve summary: a method that is right at 90 percent but badly
    wrong at 50 percent is not calibrated, and a single level would hide it.
    """
    ens, y = _check(ens, y)
    devs = []
    for lv in levels:
        alpha = (1.0 - lv) / 2.0
        lo = np.quantile(ens, alpha, axis=0)
        hi = np.quantile(ens, 1.0 - alpha, axis=0)
        devs.append(abs(float(np.mean((y >= lo) & (y <= hi))) - lv))
    return float(np.mean(devs))


def reliability_curve(
    ens: np.ndarray,
    y: np.ndarray,
    levels: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99),
) -> tuple[list[float], list[float]]:
    """(nominal, empirical) coverage pairs for a reliability plot."""
    ens, y = _check(ens, y)
    emp = []
    for lv in levels:
        alpha = (1.0 - lv) / 2.0
        lo = np.quantile(ens, alpha, axis=0)
        hi = np.quantile(ens, 1.0 - alpha, axis=0)
        emp.append(float(np.mean((y >= lo) & (y <= hi))))
    return list(levels), emp


def effective_family_count(weights: dict[str, float] | np.ndarray) -> float:
    """exp(Shannon entropy) of the family weights: the Hill number of order 1.

    Reads directly as "how many families is the evidence effectively keeping":
    1.0 means the structure is resolved to a single family, and a value near the
    bank size means the data cannot separate them at all. More interpretable
    than raw entropy in nats, which is why both are shipped.
    """
    w = np.array(list(weights.values()) if isinstance(weights, dict) else weights, dtype=float)
    w = w[w > 0]
    if w.size == 0:
        return 0.0
    w = w / w.sum()
    return float(np.exp(-np.sum(w * np.log(w))))


def parameter_dispersion(thetas: np.ndarray) -> list[float]:
    """Per-parameter coefficient of variation across ensemble members.

    The equifinality readout in parameter space: a family can predict well while
    its parameters are wildly unidentified, and this is the number that says so.
    Returns nan for a parameter whose mean is zero (no scale to normalize by).
    """
    t = np.asarray(thetas, dtype=float)
    if t.ndim != 2 or t.shape[0] < 2:
        raise ValueError("thetas must be (n_members, k) with >= 2 members")
    mean = t.mean(axis=0)
    sd = t.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(np.abs(mean) > 0, sd / np.abs(mean), np.nan)
    return [float(v) for v in cv]
