"""Bootstrap-aggregated fits of one family (rung 5 substrate).

Paired (case) bootstrap for independent observations and a moving-block bootstrap for
serially correlated data (plant time series). Aggregation by mean (bagging, Breiman
1996) or median (bragging, the E-SINDy usage: Fasel et al. 2022,
DOI 10.1098/rspa.2021.0904). The single-family bootstrap generalizes the mechanism of
Pinto et al. 2019 (DOI 10.1007/s00449-019-02181-y) beyond one hybrid structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from phenoforge.families.base import FitResult, ModelFamily
from phenoforge.fit.nls import fit_family


def bootstrap_indices(
    n: int, n_boot: int, rng: np.random.Generator, block: int | None = None
) -> np.ndarray:
    """(n_boot, n) resample index matrix; block > 1 switches to moving-block."""
    if block is None or block <= 1:
        return rng.integers(0, n, size=(n_boot, n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)
    return idx[:, :n]


@dataclass
class BootstrapEnsemble:
    """Fitted bootstrap fleet of ONE family."""

    family: ModelFamily
    fits: list[FitResult]
    kept: int
    requested: int
    meta: dict = field(default_factory=dict)

    def thetas(self) -> np.ndarray:
        return np.stack([f.theta for f in self.fits])

    def member_predictions(self, x: np.ndarray) -> np.ndarray:
        """(n_members, n_x) member prediction matrix."""
        return np.stack([self.family.predict(x, f.theta) for f in self.fits])

    def predict(self, x: np.ndarray, aggregate: str = "mean") -> np.ndarray:
        m = self.member_predictions(x)
        if aggregate == "mean":
            return m.mean(axis=0)
        if aggregate == "median":
            return np.median(m, axis=0)
        raise ValueError("aggregate must be 'mean' (bagging) or 'median' (bragging)")

    def quantiles(
        self, x: np.ndarray, qs: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    ) -> np.ndarray:
        return np.quantile(self.member_predictions(x), qs, axis=0)


def bootstrap_fit(
    family: ModelFamily,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 0,
    block: int | None = None,
    n_starts: int = 6,
) -> BootstrapEnsemble:
    """Fit `family` on n_boot bootstrap resamples of (x, y).

    Members whose fit fails (non-finite RSS) are dropped and COUNTED: `kept` vs
    `requested` is part of the record, never silently equalized.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    rng = np.random.default_rng(seed)
    idx = bootstrap_indices(n, n_boot, rng, block=block)

    fits: list[FitResult] = []
    for b in range(n_boot):
        xb, yb = x[idx[b]], y[idx[b]]
        res = fit_family(family, xb, yb, n_starts=n_starts, seed=seed + 7919 * (b + 1))
        if res.success and np.isfinite(res.rss):
            res.meta["boot_index"] = b
            fits.append(res)

    return BootstrapEnsemble(
        family=family,
        fits=fits,
        kept=len(fits),
        requested=n_boot,
        meta={"block": block, "seed": seed},
    )
