"""Bounded multistart nonlinear least squares fitting of one family.

Trust-region reflective least squares (scipy.optimize.least_squares) from multiple
random starts drawn uniformly inside the physical bounds, plus the family's declared
init. Multistart is not a luxury: several bank members (Kelsall, gamma) have
well-documented multimodal RSS surfaces on sparse batch data.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from phenoforge.families.base import FitResult, ModelFamily


def fit_family(
    family: ModelFamily,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_starts: int = 16,
    seed: int | None = 0,
    weights: np.ndarray | None = None,
    max_nfev: int = 2000,
) -> FitResult:
    """Fit one family to (x, y) and return the best-of-multistart result.

    weights, when given, scale the residuals (weighted least squares); RSS and the
    information criteria are computed on the weighted residuals so weighted fits of
    different families on the SAME data and weights stay comparable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y disagree on n")
    n = int(y.shape[0])
    lo, hi = family.bounds
    rng = np.random.default_rng(seed)

    w = None
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=float))
        if w.shape[0] != n:
            raise ValueError("weights disagree with y on n")

    def resid(theta: np.ndarray) -> np.ndarray:
        r = family.residuals(x, y, theta)
        return r * w if w is not None else r

    starts = [family.inits]
    for _ in range(max(n_starts - 1, 0)):
        starts.append(lo + rng.random(family.k) * (hi - lo))

    best: FitResult | None = None
    for theta0 in starts:
        try:
            sol = least_squares(
                resid, np.clip(theta0, lo, hi), bounds=(lo, hi), max_nfev=max_nfev
            )
        except (ValueError, FloatingPointError):
            continue
        rss = float(2.0 * sol.cost)  # least_squares cost = 0.5 * sum(r^2)
        if not np.isfinite(rss):
            continue
        if best is None or rss < best.rss:
            best = FitResult(
                family_key=family.key,
                theta=np.asarray(sol.x, dtype=float),
                rss=rss,
                n=n,
                success=bool(sol.success),
                n_starts=len(starts),
                seed=seed,
            )

    if best is None:
        # Honest failure object: init parameters, infinite RSS, success False.
        best = FitResult(
            family_key=family.key,
            theta=family.inits.copy(),
            rss=float("inf"),
            n=n,
            success=False,
            n_starts=len(starts),
            seed=seed,
        )
    return best


def fit_bank(
    families: tuple[ModelFamily, ...],
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_starts: int = 16,
    seed: int | None = 0,
    weights: np.ndarray | None = None,
) -> list[FitResult]:
    """Fit every family in a bank to the same data (rung 1/2 substrate)."""
    out: list[FitResult] = []
    for i, fam in enumerate(families):
        s = None if seed is None else seed + 1000 * i
        out.append(fit_family(fam, x, y, n_starts=n_starts, seed=s, weights=weights))
    return out
