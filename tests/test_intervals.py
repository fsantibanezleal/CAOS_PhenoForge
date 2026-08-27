"""Interval, sharpness and calibration-error metrics."""

import numpy as np
import pytest

from phenoforge.metrics import (
    coverage_deviation,
    effective_family_count,
    interval_score,
    parameter_dispersion,
    pit_calibration_error,
    reliability_curve,
    sharpness,
)


def test_interval_score_hand_value_when_observation_is_inside():
    # members {-1, 1}: the 90 percent interval is [-0.8, 0.8] by linear
    # interpolation of two order statistics; y = 0 is inside, so the score is
    # exactly the width and nothing else.
    ens = np.array([[-1.0], [1.0]])
    y = np.array([0.0])
    assert interval_score(ens, y, 0.9) == pytest.approx(sharpness(ens, 0.9))


def test_interval_score_penalises_a_miss_at_the_nominal_rate():
    ens = np.array([[-1.0], [1.0]])
    inside = interval_score(ens, np.array([0.0]), 0.9)
    outside = interval_score(ens, np.array([2.0]), 0.9)
    width = sharpness(ens, 0.9)
    hi = np.quantile(ens[:, 0], 0.95)
    assert outside == pytest.approx(width + (2.0 / 0.1) * (2.0 - hi))
    assert outside > inside


def test_widening_a_band_cannot_win_when_the_observation_is_already_inside():
    """The property that makes the interval score a proper trade-off: an
    already-covering forecast is penalised for extra width."""
    tight = np.array([[-1.0], [1.0]])
    loose = np.array([[-5.0], [5.0]])
    y = np.array([0.0])
    assert interval_score(tight, y, 0.9) < interval_score(loose, y, 0.9)


def test_sharpness_ignores_the_observation():
    ens = np.random.default_rng(0).standard_normal((200, 50))
    a = sharpness(ens, 0.9)
    b = sharpness(ens, 0.9)
    assert a == b > 0


def test_pit_calibration_error_is_small_for_a_calibrated_ensemble():
    rng = np.random.default_rng(1)
    ens = rng.standard_normal((400, 400))
    y = rng.standard_normal(400)
    assert pit_calibration_error(ens, y) < 0.1


def test_pit_calibration_error_is_large_when_the_ensemble_is_biased():
    rng = np.random.default_rng(2)
    ens = rng.standard_normal((400, 400)) + 5.0
    y = rng.standard_normal(400)
    assert pit_calibration_error(ens, y) > 0.8


def test_coverage_deviation_and_reliability_curve_agree():
    rng = np.random.default_rng(3)
    ens = rng.standard_normal((300, 300))
    y = rng.standard_normal(300)
    dev = coverage_deviation(ens, y)
    nominal, empirical = reliability_curve(ens, y)
    assert 0.0 <= dev < 0.1
    assert len(nominal) == len(empirical)
    assert all(0.0 <= e <= 1.0 for e in empirical)
    assert empirical == sorted(empirical)  # wider intervals cover at least as much


def test_effective_family_count_reads_as_a_count():
    assert effective_family_count({"a": 1.0}) == pytest.approx(1.0)
    assert effective_family_count({f"f{i}": 0.25 for i in range(4)}) == pytest.approx(4.0)
    mixed = effective_family_count({"a": 0.7, "b": 0.2, "c": 0.1})
    assert 1.0 < mixed < 3.0


def test_parameter_dispersion_flags_an_unidentified_parameter():
    rng = np.random.default_rng(4)
    tight = 2.0 + 0.01 * rng.standard_normal(200)
    loose = 2.0 + 1.50 * rng.standard_normal(200)
    cv = parameter_dispersion(np.stack([tight, loose], axis=1))
    assert cv[0] < 0.05 < cv[1]
    with pytest.raises(ValueError):
        parameter_dispersion(np.zeros((1, 2)))
