"""Hybrid grey-box with a GP discrepancy term (rung 10), pure numpy/scipy.

The Kennedy-O'Hagan decomposition: data = best phenomenological backbone + a
Gaussian-process model-discrepancy term + noise. Exact GP regression (RBF kernel
plus noise), hyperparameters by marginal-likelihood maximization; series are
short (n <= ~30) so the exact solve is trivial.

Primary sources:
- Kennedy, M.C., O'Hagan, A., 2001. Bayesian calibration of computer models.
  J. R. Stat. Soc. B 63(3), 425-464. DOI 10.1111/1467-9868.00294.
- Rasmussen, C.E., Williams, C.K.I., 2006. Gaussian Processes for Machine
  Learning. MIT Press (ch. 2 exact inference; ch. 5 marginal likelihood).
- von Stosch et al. 2014 (hybrid semi-parametric vocabulary), Comput. Chem.
  Eng. 60, 86-101.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from phenoforge.families.base import ModelFamily


def _rbf(xa: np.ndarray, xb: np.ndarray, amp2: float, ls: float) -> np.ndarray:
    d = xa[:, None] - xb[None, :]
    return amp2 * np.exp(-0.5 * (d / ls) ** 2)


@dataclass
class HybridGp:
    family: ModelFamily
    theta: np.ndarray
    x_train: np.ndarray
    resid_train: np.ndarray
    amp2: float
    ls: float
    noise2: float
    _alpha: np.ndarray
    _l_chol: np.ndarray

    def _gp_posterior(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ks = _rbf(xs, self.x_train, self.amp2, self.ls)
        mean = ks @ self._alpha
        v = np.linalg.solve(self._l_chol, ks.T)
        kss = _rbf(xs, xs, self.amp2, self.ls)
        cov = kss - v.T @ v
        return mean, cov

    def predict(self, xs: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        base = self.family.predict(xs, self.theta)
        mean, _ = self._gp_posterior(xs)
        return base + mean

    def member_draws(
        self, xs: np.ndarray, n_draws: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Posterior draws of backbone + discrepancy (function draws, no
        observation noise, consistent with the other rungs' member convention)."""
        xs = np.asarray(xs, dtype=float)
        base = self.family.predict(xs, self.theta)
        mean, cov = self._gp_posterior(xs)
        cov = cov + 1e-10 * np.eye(cov.shape[0])
        draws = rng.multivariate_normal(mean, cov, size=n_draws, method="cholesky")
        return base[None, :] + draws


def hybrid_gp_fit(
    family: ModelFamily,
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 0,  # noqa: ARG001 - deterministic; kept for the shared rung signature
) -> HybridGp:
    """Fit the GP discrepancy on the residuals of an already-fitted backbone."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    resid = y - family.predict(x, theta)
    n = x.size
    span = float(x.max() - x.min()) or 1.0
    std = float(np.std(resid)) or 1e-3

    def nll(p: np.ndarray) -> float:
        amp2, ls, noise2 = np.exp(p)
        k = _rbf(x, x, amp2, ls) + noise2 * np.eye(n)
        try:
            lchol = np.linalg.cholesky(k)
        except np.linalg.LinAlgError:
            return 1e12
        alpha = np.linalg.solve(lchol.T, np.linalg.solve(lchol, resid))
        return float(
            0.5 * resid @ alpha + np.sum(np.log(np.diag(lchol))) + 0.5 * n * np.log(2 * np.pi)
        )

    p0 = np.log([std * std + 1e-10, span / 3.0, (std * std) / 4.0 + 1e-10])
    bounds = [
        (np.log(1e-10), np.log(4.0 * std * std + 1e-8)),
        (np.log(span / 50.0), np.log(span * 3.0)),
        (np.log(1e-10), np.log(std * std + 1e-8)),
    ]
    sol = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
    amp2, ls, noise2 = np.exp(sol.x)

    k = _rbf(x, x, amp2, ls) + noise2 * np.eye(n)
    lchol = np.linalg.cholesky(k + 1e-12 * np.eye(n))
    alpha = np.linalg.solve(lchol.T, np.linalg.solve(lchol, resid))
    return HybridGp(
        family=family, theta=np.asarray(theta, dtype=float),
        x_train=x, resid_train=resid,
        amp2=float(amp2), ls=float(ls), noise2=float(noise2),
        _alpha=alpha, _l_chol=lchol,
    )
