"""The two-zone settling families must not overflow in their discarded branch.

`np.where` evaluates BOTH branches. The compression branch of the Coe-Clevenger
and Talmage-Fitch families was therefore computed for x < t_c as well, where
-(x - t_c)/tau is large and positive and exp overflows to inf, then to NaN if the
prefactor happens to be zero.

The returned values were never wrong, because the overflowed numbers are the
branch `np.where` discards. It still mattered: the warning fires inside every
fitting loop, and a warning flood down a shared output pipe is what blocked a
six-hour canonical bake for seven hours on 2026-08-29.

Clamping the exponent argument at zero is exactly equivalent where the branch is
USED, and bounds exp by 1 everywhere else. These tests pin both halves of that
claim: no warnings, and unchanged values.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phenoforge.families import thickening

TWO_ZONE = [f for f in thickening.ALL if f.key in ("sett_coe_clevenger", "sett_talmage_fitch")]


def _settling_grid() -> np.ndarray:
    return np.linspace(2.0, 300.0, 200)


@pytest.mark.parametrize("fam", TWO_ZONE, ids=lambda f: f.key)
def test_no_numerical_warning_across_the_parameter_box(fam) -> None:
    """Sweep tau down to its lower bound, which is where the overflow lived."""
    x = _settling_grid()
    lo, hi = fam.bounds
    rng = np.random.default_rng(3)
    # Only the categories that signal a real problem. UNDERFLOW is excluded on
    # purpose: exp of a large negative number going to zero is the correct answer
    # for a decaying compression term, and numpy ignores it by default, which is
    # why the bake's stderr reported overflow and invalid but never underflow.
    bad = ("overflow", "invalid value", "divide by zero")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_err = np.seterr(over="warn", invalid="warn", divide="warn", under="ignore")
        try:
            for _ in range(200):
                theta = lo + rng.random(len(lo)) * (hi - lo)
                y = fam.predict(x, theta)
                assert np.all(np.isfinite(y)), f"{fam.key}: non-finite output for theta={theta}"
            # the exact corner the overflow lived at: the smallest tau allowed
            theta = np.array([p.init for p in fam.params], dtype=float)
            theta[-1] = lo[-1]
            y = fam.predict(x, theta)
            assert np.all(np.isfinite(y))
        finally:
            np.seterr(**old_err)
    numeric = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning) and any(b in str(w.message) for b in bad)
    ]
    assert not numeric, f"{fam.key}: {len(numeric)} numerical warning(s), e.g. {numeric[0].message}"


@pytest.mark.parametrize("fam", TWO_ZONE, ids=lambda f: f.key)
def test_the_used_branch_is_unchanged_by_the_clamp(fam) -> None:
    """The clamp may only touch values np.where was going to discard.

    Recomputed here from the family's own parameters with the UNCLAMPED
    expression, restricted to the region where the compression branch is the one
    selected, and required to match to the last bit.
    """
    x = _settling_grid()
    theta = np.array([p.init for p in fam.params], dtype=float)
    y = fam.predict(x, theta)

    if fam.key == "sett_coe_clevenger":
        h0, v0, t_c, h_inf, tau = theta
        h_c = max(h0 - v0 * t_c, h_inf)
        with np.errstate(over="ignore", invalid="ignore"):
            unclamped = h_inf + (h_c - h_inf) * np.exp(-(x - t_c) / max(tau, 1e-9))
        used = x > t_c
    else:
        h0, v0, h_u, tau = theta
        t_c = max((h0 - h_u) / max(v0, 1e-12), 1e-9)
        with np.errstate(over="ignore", invalid="ignore"):
            unclamped = h_u + ((h0 - v0 * t_c) - h_u) * np.exp(-(x - t_c) / max(tau, 1e-9))
        unclamped = np.maximum(unclamped, h_u)
        used = x > t_c

    if not used.any():
        pytest.skip("default parameters never enter the compression zone on this grid")
    assert np.array_equal(y[used], unclamped[used]), (
        f"{fam.key}: the clamp changed a value in the region where the branch is used"
    )
