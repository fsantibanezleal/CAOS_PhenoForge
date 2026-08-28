"""Affine-invariant ensemble MCMC (Goodman-Weare stretch move), pure numpy.

The sampler behind rung 8 (Bayesian calibration per family). Primary sources:
- Goodman, J., Weare, J., 2010. Ensemble samplers with affine invariance.
  Communications in Applied Mathematics and Computational Science 5(1), 65-80.
  DOI 10.2140/camcos.2010.5.65.
- Foreman-Mackey, D., Hogg, D.W., Lang, D., Goodman, J., 2013. emcee: The MCMC
  Hammer. PASP 125, 306. DOI 10.1086/670067 (the reference implementation this
  compact version mirrors; reimplemented here to keep the core dependency-free
  and Pyodide-safe).

Parameterization: family parameters are sampled in an unbounded space via the
logit transform of (theta - lo)/(hi - lo), plus log-sigma for the gaussian
observation noise. The log-posterior includes the transform Jacobian, so the
prior is uniform over the physical box, and a flat prior on log-sigma within
[log(sigma_lo), log(sigma_hi)].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phenoforge.families.base import ModelFamily


def _to_unbounded(theta: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    u = (theta - lo) / (hi - lo)
    u = np.clip(u, 1e-9, 1.0 - 1e-9)
    return np.log(u / (1.0 - u))


def _to_bounded(z: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-z))
    return lo + s * (hi - lo)


def _log_jacobian(z: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    # d theta / d z = (hi-lo) * s * (1-s) with s = sigmoid(z); log|J| summed over
    # dims. Stable form: log s = -logaddexp(0, -z), log(1-s) = -logaddexp(0, z),
    # so a saturated walker (|z| large) yields -inf exactly, with no divide-by-
    # zero warning from evaluating log(0) through the naive sigmoid.
    return float(np.sum(np.log(hi - lo) - np.logaddexp(0.0, -z) - np.logaddexp(0.0, z)))


@dataclass
class PosteriorSample:
    family_key: str
    thetas: np.ndarray        # (n_samples, k) physical space
    sigmas: np.ndarray        # (n_samples,)
    log_post: np.ndarray      # (n_samples,)
    acceptance: float
    n_walkers: int
    n_steps: int


def sample_posterior(
    family: ModelFamily,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_walkers: int = 24,
    n_steps: int = 500,
    burn: int = 250,
    thin: int = 2,
    seed: int = 0,
    theta_init: np.ndarray | None = None,
    sigma_bounds: tuple[float, float] | None = None,
    stretch_a: float = 2.0,
) -> PosteriorSample:
    """Gaussian-likelihood posterior over (theta, sigma) for one family.

    `sigma_bounds` defaults to None, meaning the support of the noise-scale prior
    is derived from the spread of `y`. Pass an explicit pair only when the noise
    scale is known independently of the data; a hard-coded pair silently
    restricts which observables the sampler can represent at all.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    lo, hi = family.bounds
    k = family.k
    rng = np.random.default_rng(seed)
    # The noise scale lives in the units of y, so its prior must too. A fixed
    # pair of numbers here is only ever right for one observable: the previous
    # default of (1e-4, 0.5) was written for a recovery fraction and put the
    # walker start ABOVE the hard cap for any series with a spread over five
    # units, which made every walker start at -inf. Derived bounds cannot do
    # that: a noise standard deviation above a few times the spread of the data
    # is not a noise model, and one a millionth of it is numerically zero.
    if sigma_bounds is None:
        s = float(np.std(y))
        if not np.isfinite(s) or s <= 0.0:
            s = max(abs(float(np.mean(y))), 1.0)
        sigma_bounds = (1e-6 * s, 4.0 * s)
    slo, shi = np.log(sigma_bounds[0]), np.log(sigma_bounds[1])

    def log_post(zfull: np.ndarray) -> float:
        z, logsig = zfull[:k], zfull[k]
        if not (slo <= logsig <= shi):
            return -np.inf
        theta = _to_bounded(z, lo, hi)
        sig = np.exp(logsig)
        r = y - family.predict(x, theta)
        ll = -0.5 * n * np.log(2.0 * np.pi) - n * logsig - 0.5 * float(r @ r) / (sig * sig)
        return ll + _log_jacobian(z, lo, hi)

    center = theta_init if theta_init is not None else family.inits
    z0 = _to_unbounded(np.clip(center, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo)), lo, hi)
    # start an order of magnitude below the data spread, but never outside the
    # prior support (the previous form could and did land above the upper bound)
    sig0 = float(np.clip(np.log(np.std(y) * 0.1 + 1e-12), slo + 1e-6, shi - 1e-6))
    walkers = np.tile(np.concatenate([z0, [sig0]]), (n_walkers, 1))
    walkers += 0.1 * rng.standard_normal(walkers.shape)
    lp = np.array([log_post(w) for w in walkers])
    # revive any walker born in a -inf pocket
    for i in np.flatnonzero(~np.isfinite(lp)):
        for _ in range(50):
            walkers[i] = np.concatenate([z0, [sig0]]) + 0.05 * rng.standard_normal(k + 1)
            lp[i] = log_post(walkers[i])
            if np.isfinite(lp[i]):
                break

    dim = k + 1
    chain_thetas, chain_sigmas, chain_lp = [], [], []
    accepted = 0
    proposals = 0
    for step in range(n_steps):
        for i in range(n_walkers):
            j = rng.integers(0, n_walkers - 1)
            if j >= i:
                j += 1
            zz = ((stretch_a - 1.0) * rng.random() + 1.0) ** 2 / stretch_a
            prop = walkers[j] + zz * (walkers[i] - walkers[j])
            lp_prop = log_post(prop)
            proposals += 1
            log_ratio = (dim - 1) * np.log(zz) + lp_prop - lp[i]
            if np.log(rng.random() + 1e-300) < log_ratio:
                walkers[i] = prop
                lp[i] = lp_prop
                accepted += 1
        if step >= burn and (step - burn) % thin == 0:
            for w, lpv in zip(walkers, lp, strict=True):
                chain_thetas.append(_to_bounded(w[:k], lo, hi))
                chain_sigmas.append(np.exp(w[k]))
                chain_lp.append(lpv)

    return PosteriorSample(
        family_key=family.key,
        thetas=np.array(chain_thetas),
        sigmas=np.array(chain_sigmas),
        log_post=np.array(chain_lp),
        acceptance=accepted / max(proposals, 1),
        n_walkers=n_walkers,
        n_steps=n_steps,
    )
