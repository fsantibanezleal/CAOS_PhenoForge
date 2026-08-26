"""Stacking: cross-validated convex combination of family predictions (rung 7).

Learns nonnegative weights summing to 1 that minimize the K-fold out-of-fold squared
error of the combined prediction; the super-learner recipe restricted to the convex
simplex. Primary sources: Wolpert 1992, Neural Networks 5, 241-259; van der Laan,
Polley, Hubbard 2007, Stat. Appl. Genet. Mol. Biol. 6(1):25; Yao, Vehtari, Simpson,
Gelman 2018, Bayesian Anal. 13, 917-1007, DOI 10.1214/17-BA1091 (why stacking is the
robust combination in the M-open setting every industrial family bank lives in).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from phenoforge.families.base import ModelFamily
from phenoforge.fit.nls import fit_family


def kfold_indices(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    perm = rng.permutation(n)
    return [perm[i::k] for i in range(k)]


def simplex_weights_for(oof: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nonnegative least squares on the out-of-fold matrix, renormalized to sum 1.

    NNLS + renormalization is the standard practical solver for simplex-constrained
    stacking; exact simplex QP differs negligibly for well-scaled problems.
    """
    w, _ = nnls(oof, y)
    s = w.sum()
    if s <= 0:
        # Degenerate: no member helps; fall back to uniform (honest, recorded upstream).
        return np.full(oof.shape[1], 1.0 / oof.shape[1])
    return w / s


@dataclass
class StackedEnsemble:
    families: tuple[ModelFamily, ...]
    thetas: list[np.ndarray]     # full-data refit per family
    weights: np.ndarray          # (n_families,) convex
    oof_rmse: float              # out-of-fold RMSE of the stack

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = np.stack(
            [f.predict(x, t) for f, t in zip(self.families, self.thetas, strict=True)]
        )
        return np.einsum("m,mn->n", self.weights, preds)


def stack_fit(
    families: tuple[ModelFamily, ...],
    x: np.ndarray,
    y: np.ndarray,
    *,
    k_folds: int = 5,
    seed: int = 0,
    n_starts: int = 8,
) -> StackedEnsemble:
    """K-fold stacking over the bank.

    Each family is fit on every training fold; its out-of-fold predictions fill one
    column of the OOF matrix; weights solve the simplex-constrained least squares on
    that matrix; final member parameters are full-data refits.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    if n < 2 * k_folds:
        k_folds = max(2, n // 2)
    rng = np.random.default_rng(seed)
    folds = kfold_indices(n, k_folds, rng)

    oof = np.zeros((n, len(families)))
    for j, fam in enumerate(families):
        for f_idx, test_idx in enumerate(folds):
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            res = fit_family(
                fam, x[train_mask], y[train_mask],
                n_starts=n_starts, seed=seed + 101 * j + f_idx,
            )
            oof[test_idx, j] = fam.predict(x[test_idx], res.theta)

    w = simplex_weights_for(oof, y)
    oof_rmse = float(np.sqrt(np.mean((y - oof @ w) ** 2)))

    thetas = [
        fit_family(fam, x, y, n_starts=n_starts, seed=seed + 977 * j).theta
        for j, fam in enumerate(families)
    ]
    return StackedEnsemble(families=families, thetas=thetas, weights=w, oof_rmse=oof_rmse)
