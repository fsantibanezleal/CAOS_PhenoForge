"""Small-sample behavior: AICc is +inf for every family when the series is very
short (n <= p + 1 for all p); the bank must fall back to BIC coherently instead of
dying on an all-NaN selection (found by the Fragua sparse-n6 variant)."""

import numpy as np
import pytest

from phenoforge.ensemble import bape_fit
from phenoforge.ensemble.ic import akaike_weights, choose_criterion, select
from phenoforge.families import flotation
from phenoforge.fit import fit_bank

BANK = flotation.BATCH_FAMILIES


def _tiny_first_order(n=4, seed=0):
    t = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])[:n]
    rng = np.random.default_rng(seed)
    y = 0.85 * (1.0 - np.exp(-1.1 * t)) + 0.005 * rng.standard_normal(n)
    return t, np.clip(y, 0.0, 1.0)


def test_choose_criterion_falls_back_to_bic_on_tiny_n():
    t, y = _tiny_first_order(4)
    fits = fit_bank(BANK, t, y, n_starts=4, seed=0)
    assert all(not np.isfinite(f.aicc) for f in fits)
    assert choose_criterion(fits) == "bic"
    best = select(fits, "bic")
    assert np.isfinite(best.bic)
    w = akaike_weights(fits, "bic")
    assert w.sum() == pytest.approx(1.0)


def test_choose_criterion_prefers_aicc_when_defined():
    t = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0])
    y = 0.85 * (1.0 - np.exp(-1.1 * t))
    fits = fit_bank(BANK, t, y, n_starts=4, seed=0)
    assert choose_criterion(fits) == "aicc"


def test_bape_survives_tiny_bootstrap_resamples():
    # bootstrap resamples of a 5-point series often have <= 4 distinct points:
    # per-member BIC fallback must keep members alive
    t, y = _tiny_first_order(5)
    ens = bape_fit(BANK, t, y, n_members=40, seed=0, n_starts=3)
    assert ens.kept > 0
    shares = ens.selection_shares()
    assert sum(shares.values()) == pytest.approx(1.0)
