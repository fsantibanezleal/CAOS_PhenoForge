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
