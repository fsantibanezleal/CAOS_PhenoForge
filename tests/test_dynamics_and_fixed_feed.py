"""Process-dynamics bank and the fixed-feed projection of the energy-size laws."""

from __future__ import annotations

import numpy as np
import pytest

from phenoforge import get_family, list_families
from phenoforge.families import comminution, dynamics
from phenoforge.fit import fit_bank, fit_family


def _t() -> np.ndarray:
    return np.linspace(0.0, 12.0, 40)


@pytest.mark.parametrize("fam", dynamics.ALL, ids=lambda f: f.key)
def test_step_families_start_at_baseline_and_are_finite(fam) -> None:
    theta = np.array([p.init for p in fam.params], dtype=float)
    t = _t()
    y = fam.predict(t, theta)
    assert y.shape == t.shape
    assert np.all(np.isfinite(y))
    assert y[0] == pytest.approx(theta[0], abs=1e-9), "every family must satisfy y(0) = y_0"


@pytest.mark.parametrize("fam", dynamics.ALL, ids=lambda f: f.key)
def test_step_families_recover_their_own_parameters(fam) -> None:
    rng = np.random.default_rng(7)
    theta = np.array([p.init for p in fam.params], dtype=float)
    t = _t()
    y = fam.predict(t, theta)
    noisy = y + rng.normal(0.0, 0.002 * max(abs(theta[1]), 1.0), t.size)
    res = fit_family(fam, t, noisy, seed=3)
    assert res.success
    rmse = float(np.sqrt(res.rss / res.n))
    assert rmse < 0.05 * max(abs(theta[1]), 1.0)


def test_self_regulating_families_approach_their_gain() -> None:
    """Every family except the integrator settles at y_0 + K."""
    far = np.array([400.0])
    for fam in dynamics.ALL:
        theta = np.array([p.init for p in fam.params], dtype=float)
        y = float(fam.predict(far, theta)[0])
        if fam.key == "dyn_integrating":
            assert y > 100.0, "the integrator must keep ramping"
        else:
            assert y == pytest.approx(theta[0] + theta[1], rel=1e-6)


def test_overdamped_pair_is_continuous_at_the_repeated_pole() -> None:
    """tau1 -> tau2 is a removable singularity of the textbook closed form."""
    t = _t()
    base = np.array([0.0, 1.0, 2.0, 2.0])
    exact = dynamics.SECOND_ORDER.predict(t, base)
    reference = dynamics.CRITICALLY_DAMPED.predict(t, np.array([0.0, 1.0, 2.0]))
    assert np.allclose(exact, reference, atol=1e-9)
    for eps in (1e-3, 1e-5, 1e-7):
        near = dynamics.SECOND_ORDER.predict(t, np.array([0.0, 1.0, 2.0 + eps, 2.0]))
        assert np.all(np.isfinite(near))
        assert np.max(np.abs(near - reference)) < 1e-2


def test_underdamped_overshoots_and_the_others_do_not() -> None:
    t = np.linspace(0.0, 30.0, 600)
    over = dynamics.UNDERDAMPED.predict(t, np.array([0.0, 1.0, 1.0, 0.2]))
    assert over.max() > 1.05, "a damping ratio of 0.2 must overshoot"
    for fam in (dynamics.FIRST_ORDER, dynamics.CRITICALLY_DAMPED):
        theta = np.array([p.init for p in fam.params], dtype=float)
        y = fam.predict(t, theta)
        assert y.max() <= theta[0] + theta[1] + 1e-9


def test_dynamics_families_are_in_the_public_registry() -> None:
    keys = {f.key for f in list_families("dynamics")}
    assert keys == {f.key for f in dynamics.ALL}
    for key in keys:
        assert get_family(key).process == "dynamics"


def test_fixed_feed_projection_equals_the_two_column_call() -> None:
    f80 = 3243.0
    p80 = np.array([76.0, 101.0, 136.0, 183.0, 220.0])
    pairs = np.stack([np.full_like(p80, f80), p80], axis=-1)
    for flat, full in zip(comminution.fixed_feed_bank(f80), comminution.ALL, strict=True):
        assert flat.key == full.key
        assert flat.params == full.params
        assert flat.equation == full.equation
        theta = np.array([p.init for p in full.params], dtype=float)
        assert np.allclose(flat.predict(p80, theta), full.predict(pairs, theta))


def test_fixed_feed_bank_is_fittable_and_ranks_the_laws() -> None:
    """A Bond-generated sweep must be won by Bond in a like-for-like fit."""
    f80 = 3243.0
    p80 = np.array([76.0, 101.0, 136.0, 183.0, 220.0])
    bank = comminution.fixed_feed_bank(f80)
    truth = next(f for f in bank if f.key == "comm_bond")
    y = truth.predict(p80, np.array([18.5]))
    fits = fit_bank(bank, p80, y, seed=11)
    best = min(fits, key=lambda r: r.rss)
    assert best.family_key == "comm_bond"
    assert float(np.sqrt(best.rss / best.n)) < 1e-6


def test_fixed_feed_bank_rejects_an_impossible_feed_size() -> None:
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            comminution.fixed_feed_bank(bad)
