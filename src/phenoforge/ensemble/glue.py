"""GLUE: generalized likelihood uncertainty estimation (rung 4).

Monte Carlo sampling of parameter sets inside the physical bounds, retention of the
"behavioural" sets above a likelihood threshold, and likelihood-weighted predictive
quantiles. The equifinality readout: many parameter sets (and structures) reproduce
the data comparably well, and keeping them IS the uncertainty statement.

Primary sources: Beven and Binley 1992, Hydrol. Process. 6, 279-298; Beven 2006,
J. Hydrol. 320, 18-36, DOI 10.1016/j.jhydrol.2005.07.007; Beven and Binley 2014,
Hydrol. Process. 28, 5897-5918, DOI 10.1002/hyp.10082. GLUE weights are an informal
likelihood by construction; that caveat ships with every GLUE artifact
(Stedinger et al. 2008, DOI 10.1029/2008WR006822).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phenoforge.families.base import ModelFamily


@dataclass
class GlueEnsemble:
    family: ModelFamily
    thetas: np.ndarray          # (n_behavioural, k)
    likelihoods: np.ndarray     # (n_behavioural,) normalized to sum 1
    threshold: float
    n_sampled: int

    @property
    def n_behavioural(self) -> int:
        return int(self.thetas.shape[0])

    def member_predictions(self, x: np.ndarray) -> np.ndarray:
        return np.stack([self.family.predict(x, t) for t in self.thetas])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Likelihood-weighted mean prediction."""
        m = self.member_predictions(x)
        return np.einsum("m,mn->n", self.likelihoods, m)

    def quantiles(self, x: np.ndarray, qs: tuple[float, ...] = (0.05, 0.5, 0.95)) -> np.ndarray:
        """Likelihood-weighted predictive quantiles per x point."""
        m = self.member_predictions(x)  # (M, N)
        order = np.argsort(m, axis=0)
        sorted_m = np.take_along_axis(m, order, axis=0)
        w = self.likelihoods[order]
        cw = np.cumsum(w, axis=0)
        cw /= cw[-1:, :]
        out = np.empty((len(qs), m.shape[1]))
        for j, q in enumerate(qs):
            hit = np.argmax(cw >= q, axis=0)
            out[j] = sorted_m[hit, np.arange(m.shape[1])]
        return out


def nash_sutcliffe(y: np.ndarray, yhat: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency, the customary GLUE informal likelihood."""
    y = np.asarray(y, dtype=float)
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom == 0.0:
        return float("-inf")
    return 1.0 - float(np.sum((y - yhat) ** 2)) / denom


def glue_fit(
    family: ModelFamily,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_samples: int = 20000,
    threshold: float = 0.0,
    seed: int = 0,
) -> GlueEnsemble:
    """Sample uniform-in-bounds parameter sets; keep NSE > threshold as behavioural.

    Weights are shifted NSE (informal likelihood), normalized over the behavioural
    set. An empty behavioural set is returned EMPTY (n_behavioural == 0), which is a
    valid scientific verdict, not an error.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    lo, hi = family.bounds
    thetas = lo + rng.random((n_samples, family.k)) * (hi - lo)

    nse = np.array([nash_sutcliffe(y, family.predict(x, t)) for t in thetas])
    keep = nse > threshold
    kept = thetas[keep]
    scores = nse[keep] - threshold
    total = float(scores.sum())
    lik = scores / total if total > 0 else scores

    return GlueEnsemble(
        family=family,
        thetas=kept,
        likelihoods=lik,
        threshold=threshold,
        n_sampled=n_samples,
    )
