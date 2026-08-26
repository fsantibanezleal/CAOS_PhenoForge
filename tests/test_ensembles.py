import numpy as np
import pytest

from phenoforge.ensemble import (
    akaike_weights,
    averaged_prediction,
    bape_fit,
    bootstrap_fit,
    glue_fit,
    select,
    stack_fit,
)
from phenoforge.ensemble.bootstrap import bootstrap_indices
from phenoforge.families import flotation
from phenoforge.fit import fit_bank
from phenoforge.metrics import structural_recovery, weight_entropy


def _first_order_data(n=14, rinf=0.85, k=1.2, noise=0.008, seed=0):
    t = np.linspace(0.25, 22.0, n)
    rng = np.random.default_rng(seed)
    y = rinf * (1.0 - np.exp(-k * t)) + noise * rng.standard_normal(n)
    return t, np.clip(y, 0.0, 1.0)


BANK = flotation.BATCH_FAMILIES


def test_ic_weights_simplex_and_selection():
    t, y = _first_order_data()
    results = fit_bank(BANK, t, y, n_starts=6, seed=0)
    w = akaike_weights(results)
    assert w.shape == (len(BANK),)
    assert np.all(w >= 0.0)
    assert np.sum(w) == pytest.approx(1.0)
    best = select(results)
    assert best.family_key in {f.key for f in BANK}
    yavg, w2 = averaged_prediction(BANK, results, t)
    assert yavg.shape == t.shape
    assert np.max(np.abs(yavg - y)) < 0.1


def test_bootstrap_ensemble_shapes_and_quantiles():
    t, y = _first_order_data(seed=2)
    ens = bootstrap_fit(flotation.FIRST_ORDER, t, y, n_boot=40, seed=1, n_starts=4)
    assert 0 < ens.kept <= 40
    q = ens.quantiles(t, (0.05, 0.5, 0.95))
    assert q.shape == (3, t.size)
    assert np.all(q[0] <= q[1] + 1e-12) and np.all(q[1] <= q[2] + 1e-12)
    m_mean = ens.predict(t, "mean")
    m_med = ens.predict(t, "median")
    assert np.max(np.abs(m_mean - y)) < 0.1
    assert np.max(np.abs(m_med - y)) < 0.1


def test_block_bootstrap_indices():
    idx = bootstrap_indices(20, 5, np.random.default_rng(0), block=4)
    assert idx.shape == (5, 20)
    assert idx.max() < 20 and idx.min() >= 0


def test_glue_behavioural_and_empty_verdicts():
    t, y = _first_order_data(seed=3)
    g = glue_fit(flotation.FIRST_ORDER, t, y, n_samples=4000, threshold=0.0, seed=0)
    assert g.n_behavioural > 0
    assert g.likelihoods.sum() == pytest.approx(1.0)
    q = g.quantiles(t, (0.05, 0.5, 0.95))
    assert np.all(q[0] <= q[2])
    # An impossible threshold returns an EMPTY behavioural set, a valid verdict.
    g2 = glue_fit(flotation.FIRST_ORDER, t, y, n_samples=200, threshold=0.999999, seed=0)
    assert g2.n_behavioural == 0


def test_stacking_weights_simplex():
    t, y = _first_order_data(n=18, seed=4)
    st = stack_fit(BANK[:4], t, y, k_folds=3, seed=0, n_starts=4)
    assert st.weights.shape == (4,)
    assert np.all(st.weights >= 0.0)
    assert st.weights.sum() == pytest.approx(1.0)
    assert np.isfinite(st.oof_rmse)
    yhat = st.predict(t)
    assert np.max(np.abs(yhat - y)) < 0.15


def test_bape_structure_and_recovery():
    t, y = _first_order_data(n=16, noise=0.005, seed=5)
    ens = bape_fit(BANK, t, y, n_members=60, seed=0, n_starts=4)
    assert ens.kept > 0
    shares = ens.selection_shares()
    assert sum(shares.values()) == pytest.approx(1.0)
    incl = ens.inclusion_probabilities()
    assert set(incl) == {f.key for f in BANK}
    rec = structural_recovery(shares, "flot_first_order")
    assert rec["truth_key"] == "flot_first_order"
    assert rec["rank_of_truth"] <= 3
    q = ens.quantiles(t, (0.05, 0.5, 0.95))
    assert np.all(q[0] <= q[2])
    assert np.max(np.abs(ens.predict(t) - y)) < 0.1
    with pytest.raises(KeyError):
        structural_recovery(shares, "not_a_family")


def test_weight_entropy_bounds():
    n = 5
    uniform = {f"f{i}": 1.0 / n for i in range(n)}
    assert weight_entropy(uniform) == pytest.approx(np.log(n))
    assert weight_entropy({"a": 1.0, "b": 0.0}) == pytest.approx(0.0)
