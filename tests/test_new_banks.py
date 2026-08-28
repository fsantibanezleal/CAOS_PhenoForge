"""Shape, limit, nesting and recovery tests for the thickening, leaching,
grinding and utility banks."""

import numpy as np
import pytest

from phenoforge import get_family, list_families
from phenoforge.ensemble import bape_fit
from phenoforge.families import grinding, leaching, thickening, utility
from phenoforge.families.base import DataKind
from phenoforge.fit import fit_family
from phenoforge.metrics import structural_recovery
from phenoforge.router import route

T = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0])


def test_registry_totals_and_processes():
    fams = list_families()
    # batch flotation 7 + continuous flotation 5 + comminution 4 + grinding 4
    # + thickening 7 + leaching 6 + thermal 4 + utility 5 + dynamics 6
    assert len(fams) == 7 + 5 + 4 + 4 + 7 + 6 + 4 + 5 + 6
    assert len(list_families("thickening")) == 7
    assert len(list_families("leaching")) == 6
    assert len(list_families("utility")) == 5
    assert get_family("sett_kynch_ideal").process == "thickening"


@pytest.mark.parametrize("fam", thickening.ALL)
def test_settling_curves_monotone_decreasing_and_bounded(fam):
    h = fam.predict(T, fam.inits)
    assert h.shape == T.shape
    assert np.all(np.isfinite(h))
    assert np.all(np.diff(h) <= 1e-9), f"{fam.key} must not rise"
    assert np.all(h > 0)
    assert h[0] <= fam.inits[0] + 1e-9


@pytest.mark.parametrize("fam", leaching.ALL)
def test_conversion_curves_monotone_increasing_in_unit_interval(fam):
    t = np.array([0.5, 1.0, 4.0, 12.0, 36.0, 96.0, 240.0])
    x = fam.predict(t, fam.inits)
    assert np.all(np.isfinite(x))
    assert np.all(np.diff(x) >= -1e-9), f"{fam.key} must not decrease"
    assert np.all(x >= -1e-12) and np.all(x <= 1.0 + 1e-9)


def test_product_layer_inversion_satisfies_its_implicit_equation():
    fam = leaching.SC_PRODUCT_LAYER
    tau = 100.0
    t = np.array([1.0, 10.0, 50.0, 90.0, 99.0])
    x = fam.predict(t, np.array([1.0, tau]))
    r = 1.0 - x
    implicit = 1.0 - 3.0 * r ** (2.0 / 3.0) + 2.0 * r
    np.testing.assert_allclose(implicit, t / tau, atol=1e-9)


def test_leaching_regimes_are_ordered_at_equal_tau():
    """At EQUAL tau the three shrinking-core regimes are distinguishable by
    curvature, not by speed: product-layer control front-loads conversion (the
    ash layer is thin early, so the barrier is small) and then decelerates
    hardest, while film control is linear throughout. Hence at t/tau = 0.1 the
    ordering is film < reaction < product, and all three reach X_inf at t = tau.
    This is the shape signature the ensemble discriminates on."""
    theta = np.array([1.0, 100.0])
    early = np.array([10.0])
    film = leaching.SC_FILM.predict(early, theta)[0]
    reaction = leaching.SC_REACTION.predict(early, theta)[0]
    product = leaching.SC_PRODUCT_LAYER.predict(early, theta)[0]
    assert film < reaction < product
    assert film == pytest.approx(0.1)

    at_tau = np.array([100.0])
    for fam in (leaching.SC_FILM, leaching.SC_REACTION, leaching.SC_PRODUCT_LAYER):
        assert fam.predict(at_tau, theta)[0] == pytest.approx(1.0, abs=1e-6)


def test_grinding_nesting_limits():
    t = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    base = grinding.AUSTIN_FIRST_ORDER.predict(t, np.array([1.0, 0.4]))
    rollover = grinding.AUSTIN_NON_FIRST_ORDER.predict(t, np.array([1.0, 0.4, 1.0]))
    np.testing.assert_allclose(rollover, base, rtol=1e-10)
    plateau = grinding.WHITEN_PLATEAU.predict(t, np.array([1.0, 0.4, 0.0]))
    np.testing.assert_allclose(plateau, base, rtol=1e-10)


def test_utility_power_law_nests_proportional():
    tonnage = np.array([1.0, 5.0, 20.0, 50.0])
    prop = utility.PER_TONNE.predict(tonnage, np.array([3.0]))
    power = utility.POWER_LAW_SCALING.predict(tonnage, np.array([3.0, 1.0]))
    np.testing.assert_allclose(power, prop, rtol=1e-10)


def test_router_separates_the_new_banks():
    sett = {f.key for f in route((DataKind.SETTLING_CURVE,))}
    leach = {f.key for f in route((DataKind.CONVERSION_TIME,))}
    grind = {f.key for f in route((DataKind.PSD_TIME,))}
    util = {f.key for f in route((DataKind.ANNUAL_BALANCE,))}
    assert len(sett) == 7 and all(k.startswith("sett_") for k in sett)
    assert len(leach) == 6 and all(k.startswith("leach_") for k in leach)
    assert len(grind) == 4 and all(k.startswith("grind_") for k in grind)
    assert len(util) == 5 and all(k.startswith("util_") for k in util)
    assert not (sett & leach & grind & util)


def test_fit_recovers_settling_parameters():
    truth = np.array([0.40, 0.006, 0.09])
    t = np.array([1.0, 3.0, 6.0, 10.0, 20.0, 35.0, 55.0, 80.0])
    y = thickening.KYNCH_IDEAL.predict(t, truth)
    res = fit_family(thickening.KYNCH_IDEAL, t, y, n_starts=12, seed=0)
    assert res.success
    np.testing.assert_allclose(res.theta, truth, rtol=0.05)


def test_bape_recovers_the_leaching_truth_family():
    truth = leaching.SC_REACTION
    theta = np.array([0.9, 60.0])
    t = np.array([2.0, 5.0, 10.0, 18.0, 30.0, 45.0, 60.0, 80.0, 110.0])
    rng = np.random.default_rng(0)
    y = np.clip(truth.predict(t, theta) + 0.004 * rng.standard_normal(t.size), 0.0, 1.0)
    ens = bape_fit(leaching.ALL, t, y, n_members=80, seed=1, n_starts=4)
    rec = structural_recovery(ens.selection_shares(), truth.key)
    assert rec["rank_of_truth"] <= 2
