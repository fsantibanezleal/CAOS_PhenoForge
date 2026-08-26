import numpy as np

from phenoforge.families import flotation
from phenoforge.fit import fit_bank, fit_family


def _first_order_data(rinf=0.85, k=1.2, noise=0.0, seed=0):
    t = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0])
    rng = np.random.default_rng(seed)
    y = rinf * (1.0 - np.exp(-k * t)) + noise * rng.standard_normal(t.size)
    return t, y


def test_recovers_first_order_parameters_clean():
    t, y = _first_order_data()
    res = fit_family(flotation.FIRST_ORDER, t, y, n_starts=8, seed=1)
    assert res.success
    np.testing.assert_allclose(res.theta, [0.85, 1.2], rtol=1e-4)


def test_recovers_under_noise():
    t, y = _first_order_data(noise=0.01, seed=3)
    res = fit_family(flotation.FIRST_ORDER, t, y, n_starts=8, seed=1)
    assert res.success
    np.testing.assert_allclose(res.theta, [0.85, 1.2], rtol=0.15)


def test_information_criteria_finite_and_ordered():
    t, y = _first_order_data(noise=0.01, seed=5)
    res = fit_family(flotation.FIRST_ORDER, t, y, seed=2)
    assert np.isfinite(res.aic) and np.isfinite(res.bic) and np.isfinite(res.aicc)
    assert res.aicc >= res.aic


def test_aicc_inf_when_overparameterized():
    # 4 free params + variance vs n=5 points: correction denominator <= 0 -> +inf.
    t = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    y = 0.8 * (1.0 - np.exp(-t))
    res = fit_family(flotation.KELSALL_MOD, t, y, n_starts=4, seed=0)
    assert res.aicc == float("inf")


def test_fit_bank_returns_one_result_per_family():
    t, y = _first_order_data(noise=0.005, seed=7)
    results = fit_bank(flotation.BATCH_FAMILIES, t, y, n_starts=6, seed=0)
    assert len(results) == len(flotation.BATCH_FAMILIES)
    assert {r.family_key for r in results} == {f.key for f in flotation.BATCH_FAMILIES}


def test_weighted_fit_runs():
    t, y = _first_order_data(noise=0.01, seed=9)
    w = np.linspace(1.0, 2.0, t.size)
    res = fit_family(flotation.FIRST_ORDER, t, y, weights=w, seed=0)
    assert res.success
