"""Structural-uncertainty readouts over family-level ensembles."""

from __future__ import annotations

import numpy as np


def weight_entropy(weights: dict[str, float] | np.ndarray) -> float:
    """Shannon entropy (nats) of a family weight/share distribution.

    0 means one family dominates (structure resolved); log(n_families) means the data
    cannot distinguish the families (structural equifinality).
    """
    w = np.array(list(weights.values()) if isinstance(weights, dict) else weights, dtype=float)
    w = w[w > 0]
    if w.size == 0:
        return 0.0
    w = w / w.sum()
    return float(-np.sum(w * np.log(w)))


def structural_recovery(shares: dict[str, float], truth_key: str) -> dict[str, float | bool | str]:
    """Score a known-truth (synthetic) case: did the ensemble put its mass on the
    generating family?

    Returns a record, never a bare bool, so a missing truth key is an explicit error
    and the score can never silently run as null (the SymLab truth=None lesson).
    """
    if truth_key not in shares:
        raise KeyError(
            f"truth family '{truth_key}' is not in the ensemble's bank: "
            f"{sorted(shares)}; structural recovery CANNOT be scored"
        )
    ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
    top_key, top_share = ranked[0]
    return {
        "truth_key": truth_key,
        "truth_share": float(shares[truth_key]),
        "top_key": top_key,
        "top_share": float(top_share),
        "recovered_top1": bool(top_key == truth_key),
        "rank_of_truth": int([k for k, _ in ranked].index(truth_key) + 1),
    }
