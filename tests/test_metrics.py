import numpy as np
import pytest

from phenoforge.metrics import (
    crps_ensemble,
    interval_coverage,
    mae,
    pit_histogram,
    pit_values,
    r2,
    rmse,
)


def test_point_metrics():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.0, 2.0, 4.0])
    assert rmse(y, yhat) == pytest.approx(np.sqrt(1.0 / 3.0))
    assert mae(y, yhat) == pytest.approx(1.0 / 3.0)
    assert r2(y, y) == pytest.approx(1.0)


def test_crps_two_member_hand_value():
    # y = 0; members {-1, 1}: term1 = 1, spread E|X-X'| = (0+2+2+0)/4 = 1 -> crps = 0.5
    ens = np.array([[-1.0], [1.0]])
    y = np.array([0.0])
    assert crps_ensemble(ens, y) == pytest.approx(0.5)


def test_crps_of_near_degenerate_ensemble_approaches_mae():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(50)
    point = y + 0.3
    ens = np.stack([point, point + 1e-9])
    assert crps_ensemble(ens, y) == pytest.approx(0.3, rel=1e-6)


def test_pit_and_coverage():
    rng = np.random.default_rng(1)
    ens = rng.standard_normal((200, 100))
    y = rng.standard_normal(100)
    p = pit_values(ens, y)
    assert np.all((p >= 0.0) & (p <= 1.0))
    cov = interval_coverage(ens, y, 0.9)
    assert 0.8 <= cov <= 1.0
    counts, edges = pit_histogram(ens, y, bins=10)
    assert counts.sum() == 100
    assert edges[0] == 0.0 and edges[-1] == 1.0


def test_calibration_input_validation():
    with pytest.raises(ValueError):
        crps_ensemble(np.zeros((1, 5)), np.zeros(5))
    with pytest.raises(ValueError):
        interval_coverage(np.zeros((3, 5)), np.zeros(5), 1.5)
