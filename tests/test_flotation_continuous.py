"""Continuous (plant) flotation bank: limits, nesting and identifiability."""

import numpy as np
import pytest

from phenoforge.ensemble import bape_fit
from phenoforge.families import flotation
from phenoforge.families import flotation_continuous as fc
from phenoforge.families.base import DataKind
from phenoforge.fit import fit_family
from phenoforge.metrics import structural_recovery
from phenoforge.router import route

TAU = np.array([0.5, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0])


@pytest.mark.parametrize("fam", fc.ALL)
def test_continuous_recovery_is_monotone_and_bounded(fam):
    r = fam.predict(TAU, fam.inits)
    assert r.shape == TAU.shape
    assert np.all(np.isfinite(r))
    assert np.all(np.diff(r) >= -1e-12), f"{fam.key} must not fall with residence time"
    assert np.all(r >= 0.0) and np.all(r <= 1.0 + 1e-9)


def test_n_mixers_nests_both_bounds():
    """N = 1 IS the single perfect mixer; N to infinity approaches plug flow.
    The nesting is what makes the fitted N a measurement rather than a knob."""
    theta_common = (0.9, 0.4)
    one = fc.N_MIXERS.predict(TAU, np.array([*theta_common, 1.0]))
    mixer = fc.PERFECT_MIXER.predict(TAU, np.array(theta_common))
    np.testing.assert_allclose(one, mixer, rtol=1e-10)

    many = fc.N_MIXERS.predict(TAU, np.array([*theta_common, 5000.0]))
    plug = fc.PLUG_FLOW.predict(TAU, np.array(theta_common))
    np.testing.assert_allclose(many, plug, rtol=2e-3)


def test_plug_flow_dominates_a_single_mixer_at_equal_tau():
    """The RTD ordering that motivates the bank: at the same mean residence time
    a plug-flow vessel recovers more than one perfectly mixed vessel, because
    the exponential RTD sends part of the feed straight through."""
    theta = np.array([0.9, 0.4])
    plug = fc.PLUG_FLOW.predict(TAU, theta)
    mixer = fc.PERFECT_MIXER.predict(TAU, theta)
    assert np.all(plug > mixer)


def test_two_class_reduces_to_single_mixer_when_classes_coincide():
    single = fc.PERFECT_MIXER.predict(TAU, np.array([0.9, 0.5]))
    both = fc.TWO_CLASS.predict(TAU, np.array([0.9, 0.4, 0.5, 0.5]))
    np.testing.assert_allclose(both, single, rtol=1e-10)


def test_batch_and_continuous_banks_are_routed_apart():
    batch = {f.key for f in route((DataKind.TIMED_RECOVERY,))}
    cont = {f.key for f in route((DataKind.CONTINUOUS_RECOVERY,))}
    assert len(batch) == len(flotation.BATCH_FAMILIES)
    assert cont == {f.key for f in fc.ALL}
    assert not (batch & cont)


def test_fit_recovers_continuous_parameters_under_noise():
    truth = np.array([0.88, 0.35])
    rng = np.random.default_rng(0)
    y = fc.PERFECT_MIXER.predict(TAU, truth) + 0.006 * rng.standard_normal(TAU.size)
    res = fit_family(fc.PERFECT_MIXER, TAU, y, n_starts=12, seed=1)
    assert res.success
    np.testing.assert_allclose(res.theta, truth, rtol=0.15)


def test_bape_ranks_the_generating_continuous_family_highly():
    truth = fc.TWO_CLASS
    theta = np.array([0.92, 0.45, 2.0, 0.05])
    tau = np.array([0.4, 0.8, 1.5, 2.5, 4.0, 6.0, 9.0, 13.0, 18.0, 25.0])
    rng = np.random.default_rng(2)
    y = np.clip(truth.predict(tau, theta) + 0.004 * rng.standard_normal(tau.size), 0.0, 1.0)
    ens = bape_fit(fc.ALL, tau, y, n_members=90, seed=3, n_starts=5)
    rec = structural_recovery(ens.selection_shares(), truth.key)
    assert rec["rank_of_truth"] <= 2
