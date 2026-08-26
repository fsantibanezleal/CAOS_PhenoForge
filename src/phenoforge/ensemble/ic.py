"""Information-criterion selection and model averaging across a family bank.

Rungs 2 and 3 of the Fragua ladder:
- select(): the better current practice (AICc/BIC winner-take-all).
- akaike_weights() + averaged_prediction(): keep the WHOLE family and average with
  IC weights (multimodel inference, Burnham and Anderson 2004,
  DOI 10.1177/0049124104268644).
"""

from __future__ import annotations

import numpy as np

from phenoforge.families.base import FitResult, ModelFamily


def _criterion_values(results: list[FitResult], criterion: str) -> np.ndarray:
    if criterion not in ("aic", "aicc", "bic"):
        raise ValueError("criterion must be one of aic, aicc, bic")
    return np.array([getattr(r, criterion) for r in results], dtype=float)


def select(results: list[FitResult], criterion: str = "aicc") -> FitResult:
    """Winner-take-all selection by an information criterion."""
    vals = _criterion_values(results, criterion)
    if not np.isfinite(vals).any():
        raise ValueError("no finite criterion value in the bank (all fits failed)")
    return results[int(np.nanargmin(np.where(np.isfinite(vals), vals, np.nan)))]


def akaike_weights(results: list[FitResult], criterion: str = "aicc") -> np.ndarray:
    """Normalized exp(-delta/2) weights over the bank.

    Non-finite criteria (failed or over-parameterized fits) receive weight 0; the
    remainder renormalizes. This is the honest treatment: an unfittable member does
    not silently disappear from the record, it carries zero weight.
    """
    vals = _criterion_values(results, criterion)
    finite = np.isfinite(vals)
    w = np.zeros_like(vals)
    if finite.any():
        d = vals[finite] - np.min(vals[finite])
        e = np.exp(-0.5 * d)
        w[finite] = e / np.sum(e)
    return w


def averaged_prediction(
    families: tuple[ModelFamily, ...],
    results: list[FitResult],
    x: np.ndarray,
    criterion: str = "aicc",
) -> tuple[np.ndarray, np.ndarray]:
    """IC-weight model-averaged prediction over the bank.

    Returns (y_avg, weights) with weights aligned to `families`/`results` order.
    """
    if len(families) != len(results):
        raise ValueError("families and results disagree in length")
    w = akaike_weights(results, criterion)
    preds = np.stack(
        [fam.predict(x, res.theta) for fam, res in zip(families, results, strict=True)]
    )
    return np.einsum("m,mn->n", w, preds), w
