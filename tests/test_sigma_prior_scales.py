"""The Bayesian noise-scale prior must live in the units of the data.

`sample_posterior` defaulted to sigma_bounds=(1e-4, 0.5) in RAW target units,
which is right for a recovery fraction and meaningless for anything else. It was
not merely tight: the walker start is an order of magnitude below the data
spread, so for a series with a 17 MW spread the start sat ABOVE the hard cap and
every walker was born at -inf. That rung is the one the results currently call
the best calibrated.

This is the third instance of one pattern (deep-ensemble sigmoid, E-SINDy
blow-up bound, this), so the tests below are written against the pattern rather
than the instance: a sampler handed a large-scale observable must produce a
posterior whose noise scale matches the residuals it actually sees.
"""

from __future__ import annotations

import numpy as np
import pytest

from phenoforge.bayes.gw import sample_posterior
from phenoforge.families import thermal


def _megawatt_series(sigma: float = 4.0, seed: int = 5):
    """A power-derating series: the shape that broke the fixed prior."""
    fam = thermal.ALL[0]
    theta = np.array([p.init for p in fam.params], dtype=float)
    t = np.linspace(2.0, 37.0, 30)
    clean = fam.predict(t, theta)
    rng = np.random.default_rng(seed)
    return fam, t, clean + sigma * rng.standard_normal(t.size), theta


def test_a_megawatt_series_produces_a_live_posterior() -> None:
    """With the old fixed cap every walker started at -inf on this input."""
    fam, t, y, _ = _megawatt_series()
    post = sample_posterior(fam, t, y, n_walkers=16, n_steps=200, burn=100, seed=3)
    draws = np.asarray(post.thetas)
    assert draws.ndim == 2 and draws.shape[0] >= 10
    assert np.all(np.isfinite(draws))
    # a degenerate sampler returns the same point over and over
    assert float(draws.std(axis=0).max()) > 0.0, "the posterior did not move"


def test_the_recovered_noise_scale_matches_the_data() -> None:
    """The prior must be able to REACH the true noise level, not clamp under it."""
    fam, t, y, _ = _megawatt_series(sigma=4.0)
    post = sample_posterior(fam, t, y, n_walkers=20, n_steps=400, burn=200, seed=7)
    sig = np.asarray(post.sigmas, dtype=float)
    assert np.all(np.isfinite(sig))
    med = float(np.median(sig))
    assert 1.0 < med < 16.0, (
        f"median posterior sigma {med:.3g} for a series with a true noise sd of 4.0; "
        "a prior capped in the units of some other observable would pin this near its bound"
    )


def test_a_unit_interval_series_still_works() -> None:
    """The original observable must not regress while the prior is generalized."""
    from phenoforge.families import flotation

    fam = flotation.BATCH_FAMILIES[0]
    theta = np.array([p.init for p in fam.params], dtype=float)
    t = np.linspace(0.5, 12.0, 20)
    rng = np.random.default_rng(11)
    y = np.clip(fam.predict(t, theta) + 0.01 * rng.standard_normal(t.size), 0.0, 1.0)
    post = sample_posterior(fam, t, y, n_walkers=16, n_steps=300, burn=150, seed=2)
    sig = np.asarray(post.sigmas, dtype=float)
    assert 0.001 < float(np.median(sig)) < 0.2


def test_an_explicit_prior_is_still_honoured() -> None:
    """A caller who knows the noise scale independently may still pin it."""
    fam, t, y, _ = _megawatt_series()
    post = sample_posterior(
        fam, t, y, n_walkers=16, n_steps=200, burn=100, seed=3,
        sigma_bounds=(1.0, 2.0),
    )
    sig = np.asarray(post.sigmas, dtype=float)
    assert sig.min() >= 1.0 - 1e-9 and sig.max() <= 2.0 + 1e-9


def test_a_constant_series_does_not_collapse_the_prior() -> None:
    """Zero spread must not produce a zero-width or inverted support."""
    fam = thermal.ALL[0]
    t = np.linspace(2.0, 37.0, 20)
    y = np.full(t.size, 450.0)
    post = sample_posterior(fam, t, y, n_walkers=12, n_steps=120, burn=60, seed=1)
    assert np.all(np.isfinite(np.asarray(post.sigmas, dtype=float)))


@pytest.mark.parametrize("scale", [1.0, 1e2, 1e4])
def test_the_posterior_follows_the_scale_of_the_observable(scale: float) -> None:
    """Rescaling the observable must rescale the recovered noise, not clamp it."""
    fam, t, y, _ = _megawatt_series(sigma=4.0)
    post = sample_posterior(
        fam, t, y * scale, n_walkers=16, n_steps=250, burn=125, seed=4
    )
    med = float(np.median(np.asarray(post.sigmas, dtype=float)))
    assert med > 0.05 * scale, (
        f"scale {scale:g}: median sigma {med:.4g} is pinned far below the data spread"
    )
