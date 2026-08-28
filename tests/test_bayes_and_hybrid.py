import numpy as np
import pytest

from phenoforge.bayes import bic_model_probabilities, bma_fit, sample_posterior
from phenoforge.families import flotation
from phenoforge.fit import fit_bank
from phenoforge.hybrid import hybrid_gp_fit

BANK = flotation.BATCH_FAMILIES


def _data(n=12, rinf=0.85, k=1.2, noise=0.01, seed=0):
    t = np.linspace(0.25, 20.0, n)
    rng = np.random.default_rng(seed)
    y = rinf * (1.0 - np.exp(-k * t)) + noise * rng.standard_normal(n)
    return t, np.clip(y, 0.0, 1.0)


def test_posterior_brackets_truth_and_accepts():
    t, y = _data()
    post = sample_posterior(
        flotation.FIRST_ORDER, t, y, n_walkers=16, n_steps=300, burn=150, seed=1
    )
    assert post.thetas.shape[0] > 100
    assert 0.05 < post.acceptance < 0.95
    med = np.median(post.thetas, axis=0)
    assert med[0] == pytest.approx(0.85, abs=0.05)
    assert med[1] == pytest.approx(1.2, rel=0.25)
    lo, hi = flotation.FIRST_ORDER.bounds
    assert np.all(post.thetas >= lo) and np.all(post.thetas <= hi)
    assert np.all(post.sigmas > 0)


def test_bic_probabilities_simplex():
    t, y = _data(seed=2)
    fits = fit_bank(BANK, t, y, n_starts=6, seed=0)
    p = bic_model_probabilities(fits)
    assert p.shape == (len(BANK),)
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p >= 0)


def test_bma_mixture_draws():
    t, y = _data(seed=3)
    fits = fit_bank(BANK, t, y, n_starts=6, seed=0)
    ens = bma_fit(BANK, t, y, fits, n_walkers=14, n_steps=160, burn=80, seed=0)
    assert ens.model_probs.sum() == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    curves, fam_idx = ens.member_draws(t, 60, rng)
    assert curves.shape == (60, t.size)
    assert np.max(np.abs(curves.mean(axis=0) - y)) < 0.15
    w = ens.family_weights()
    assert sum(w.values()) == pytest.approx(1.0)


def test_hybrid_gp_reduces_misspecification_residual():
    # backbone deliberately misspecified: fully-mixed fitted to first-order data
    t, y = _data(n=14, noise=0.005, seed=4)
    from phenoforge.fit import fit_family

    res = fit_family(flotation.FULLY_MIXED, t, y, n_starts=8, seed=0)
    hyb = hybrid_gp_fit(flotation.FULLY_MIXED, res.theta, t, y)
    resid_backbone = float(np.sqrt(np.mean((y - flotation.FULLY_MIXED.predict(t, res.theta)) ** 2)))
    resid_hybrid = float(np.sqrt(np.mean((y - hyb.predict(t)) ** 2)))
    assert resid_hybrid < resid_backbone
    rng = np.random.default_rng(0)
    draws = hyb.member_draws(np.linspace(0.5, 25.0, 40), 30, rng)
    assert draws.shape == (30, 40)
    assert np.all(np.isfinite(draws))


def test_log_jacobian_stable_at_saturation():
    """A saturated walker (|z| large) must yield the exact finite log-Jacobian
    (log(hi-lo) - |z| per dim, to leading order) and do so silently; the naive
    sigmoid form both emitted divide-by-zero RuntimeWarnings through log(1-s)
    at s == 1.0 and collapsed the value to -inf (observed across whole
    canonical bakes)."""
    from phenoforge.bayes.gw import _log_jacobian

    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 5.0])
    z_mod = np.array([0.3, -1.7])
    s = 1.0 / (1.0 + np.exp(-z_mod))
    naive = float(np.sum(np.log(hi - lo) + np.log(s) + np.log(1.0 - s)))
    assert _log_jacobian(z_mod, lo, hi) == pytest.approx(naive, rel=1e-12)

    with np.errstate(divide="raise"):  # the naive form would raise here
        val = _log_jacobian(np.array([800.0, -800.0]), lo, hi)
    assert val == pytest.approx(np.log(5.0) - 1600.0, rel=1e-12)


def test_hybrid_gp_survives_a_numerically_singular_kernel():
    """An RBF kernel on a large-valued, closely-spaced driver (an eleven-year
    record indexed 1..11 with values near 1.3e4) is numerically singular even
    though it is positive definite in exact arithmetic. Plain Cholesky raised
    LinAlgError and killed a canonical bake; escalating jitter must recover it."""
    x = np.arange(1.0, 12.0)
    y = np.array([12951.0, 13071.6, 13614.2, 13263.5, 13357.5, 12450.0,
                  12089.3, 11843.6, 11918.0, 12016.8, 14984.1])
    from phenoforge.families import utility
    from phenoforge.fit import fit_family

    res = fit_family(utility.EXP_TREND, x, y, n_starts=8, seed=0)
    hyb = hybrid_gp_fit(utility.EXP_TREND, res.theta, x, y)
    assert np.all(np.isfinite(hyb.predict(x)))
    draws = hyb.member_draws(x, 20, np.random.default_rng(0))
    assert draws.shape == (20, 11) and np.all(np.isfinite(draws))
