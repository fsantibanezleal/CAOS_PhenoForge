"""Probabilistic calibration metrics for ensemble predictions.

The ensemble differentiator: a single best-fit model has no honest predictive
distribution; these metrics score the one the ensemble provides.

- CRPS (empirical, ensemble form): Gneiting and Raftery 2007, J. Am. Stat. Assoc.
  102, 359-378, DOI 10.1198/016214506000001437. crps = mean|X - y| - 0.5 mean|X - X'|.
- PIT (probability integral transform) values: uniform under perfect calibration.
- Central-interval coverage vs nominal.

All functions take a member-prediction matrix `ens` with shape (n_members, n_points)
and observations `y` with shape (n_points,).
"""

from __future__ import annotations

import numpy as np


def _check(ens: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ens = np.asarray(ens, dtype=float)
    y = np.asarray(y, dtype=float)
    if ens.ndim != 2 or y.ndim != 1 or ens.shape[1] != y.shape[0]:
        raise ValueError("ens must be (n_members, n_points) and y (n_points,)")
    if ens.shape[0] < 2:
        raise ValueError("need >= 2 ensemble members")
    return ens, y


def crps_ensemble(ens: np.ndarray, y: np.ndarray) -> float:
    """Mean empirical CRPS over the points (lower is better; units of y)."""
    ens, y = _check(ens, y)
    m = ens.shape[0]
    term1 = np.mean(np.abs(ens - y[None, :]), axis=0)
    # Pairwise member spread via the sorted-representation identity, O(M log M) per point.
    s = np.sort(ens, axis=0)
    i = np.arange(1, m + 1)[:, None]
    term2 = 2.0 * np.sum(s * (2.0 * i - m - 1.0), axis=0) / (m * m)
    return float(np.mean(term1 - 0.5 * term2))


def pit_values(ens: np.ndarray, y: np.ndarray) -> np.ndarray:
    """PIT per point: rank of y within the ensemble, randomized at ties on the
    half-open convention (fraction of members strictly below plus half of exact ties)."""
    ens, y = _check(ens, y)
    below = np.mean(ens < y[None, :], axis=0)
    ties = np.mean(ens == y[None, :], axis=0)
    return below + 0.5 * ties


def interval_coverage(ens: np.ndarray, y: np.ndarray, level: float = 0.9) -> float:
    """Empirical coverage of the central `level` interval (target: == level)."""
    ens, y = _check(ens, y)
    if not (0.0 < level < 1.0):
        raise ValueError("level in (0, 1)")
    alpha = (1.0 - level) / 2.0
    lo = np.quantile(ens, alpha, axis=0)
    hi = np.quantile(ens, 1.0 - alpha, axis=0)
    return float(np.mean((y >= lo) & (y <= hi)))


def pit_histogram(ens: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """(counts, edges) of the PIT histogram; flat under perfect calibration."""
    p = pit_values(ens, y)
    counts, edges = np.histogram(p, bins=bins, range=(0.0, 1.0))
    return counts, edges
