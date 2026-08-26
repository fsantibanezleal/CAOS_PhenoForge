"""Bayesian model averaging across a family bank (rung 8 aggregation).

Posterior model probabilities via the BIC approximation to Bayes factors
(Raftery, A.E., 1995. Bayesian model selection in social research. Sociological
Methodology 25, 111-163, DOI 10.2307/271063; the standard practical route when
exact marginal likelihoods are out of reach), over equal model priors. The
predictive is a mixture: per-family posterior predictive draws (from the
Goodman-Weare chains) resampled proportionally to the model probabilities.

Contrast intent (the M-open exhibit, plan claim 3): BMA weights concentrate as n
grows even when EVERY family is wrong; Bayesian stacking is the robust
alternative there (Yao, Vehtari, Simpson, Gelman 2018, DOI 10.1214/17-BA1091).
Both are shipped so the failure is demonstrable, not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phenoforge.bayes.gw import PosteriorSample, sample_posterior
from phenoforge.families.base import FitResult, ModelFamily


def bic_model_probabilities(results: list[FitResult]) -> np.ndarray:
    """Equal-prior posterior model probabilities from per-family BIC."""
    bic = np.array([r.bic for r in results], dtype=float)
    ok = np.isfinite(bic)
    p = np.zeros_like(bic)
    if ok.any():
        d = bic[ok] - np.min(bic[ok])
        e = np.exp(-0.5 * d)
        p[ok] = e / e.sum()
    return p


@dataclass
class BmaEnsemble:
    families: tuple[ModelFamily, ...]
    posteriors: list[PosteriorSample]
    model_probs: np.ndarray

    def member_draws(
        self, x: np.ndarray, n_draws: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """(curves (n_draws, n_x), family_index_per_draw). Mixture draws: family by
        model probability, theta from that family's chain."""
        fam_idx = rng.choice(len(self.families), size=n_draws, p=self.model_probs)
        curves = np.empty((n_draws, np.asarray(x).shape[0]))
        for d, fi in enumerate(fam_idx):
            post = self.posteriors[fi]
            t = post.thetas[rng.integers(0, post.thetas.shape[0])]
            curves[d] = self.families[fi].predict(x, t)
        return curves, fam_idx

    def family_weights(self) -> dict[str, float]:
        return {
            f.key: float(p)
            for f, p in zip(self.families, self.model_probs, strict=True)
        }


def bma_fit(
    families: tuple[ModelFamily, ...],
    x: np.ndarray,
    y: np.ndarray,
    fits: list[FitResult],
    *,
    n_walkers: int = 20,
    n_steps: int = 400,
    burn: int = 200,
    seed: int = 0,
    min_prob: float = 5e-3,
) -> BmaEnsemble:
    """Sample posteriors for every family whose model probability exceeds
    `min_prob` (the rest keep probability mass but contribute no draws only if
    truly negligible; probabilities are renormalized over the sampled set so the
    mixture stays proper)."""
    probs = bic_model_probabilities(fits)
    sampled: list[PosteriorSample | None] = []
    keep = probs >= min_prob
    if not keep.any():
        keep = probs == probs.max()
    for j, (fam, fit) in enumerate(zip(families, fits, strict=True)):
        if keep[j]:
            sampled.append(
                sample_posterior(
                    fam, x, y,
                    n_walkers=n_walkers, n_steps=n_steps, burn=burn,
                    seed=seed + 37 * (j + 1), theta_init=fit.theta,
                )
            )
        else:
            sampled.append(None)
    fams = tuple(f for f, k in zip(families, keep, strict=True) if k)
    posts = [s for s in sampled if s is not None]
    p = probs[keep]
    p = p / p.sum()
    return BmaEnsemble(families=fams, posteriors=posts, model_probs=p)
