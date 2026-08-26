"""BAPE: Bootstrap-Aggregated Phenomenological Ensembles (rung 6, the novel method).

The random-forest recipe with phenomenological model families as base learners:

- Each ensemble MEMBER sees (a) a bootstrap resample of the data (the bagging axis,
  Breiman 1996) and (b) a random SUBSET of the family library (the feature-subsampling
  analog applied to equation structure; in E-SINDy this is "library bagging" over
  candidate terms, Fasel et al. 2022, DOI 10.1098/rspa.2021.0904; here it operates
  over whole curated families).
- Within its subset, the member fits every family and keeps the AICc-best fit (one
  member = one fitted phenomenological model, the way one random-forest member is one
  tree).
- The ensemble aggregates member predictions (mean/median/quantiles) and reads
  STRUCTURE off the member population: the inclusion probability of a family is the
  fraction of members that selected it (the E-SINDy inclusion-probability readout
  lifted from terms to families).

Differentiation from prior art (all cited in the package docstring): E-SINDy bags
sparse regressions over generic term libraries; Pinto et al. 2019 bags ONE hybrid
structure; BMA/stacking weight given models without resampling; GLUE keeps parameter
sets of given structures. BAPE crosses the family axis with the bootstrap axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from phenoforge.ensemble.bootstrap import bootstrap_indices
from phenoforge.families.base import FitResult, ModelFamily
from phenoforge.fit.nls import fit_family


@dataclass
class BapeMember:
    family: ModelFamily
    fit: FitResult
    boot_index: int
    subset_keys: tuple[str, ...]


@dataclass
class BapeEnsemble:
    families: tuple[ModelFamily, ...]
    members: list[BapeMember]
    requested: int
    meta: dict = field(default_factory=dict)

    @property
    def kept(self) -> int:
        return len(self.members)

    def member_predictions(self, x: np.ndarray) -> np.ndarray:
        return np.stack([m.family.predict(x, m.fit.theta) for m in self.members])

    def predict(self, x: np.ndarray, aggregate: str = "mean") -> np.ndarray:
        m = self.member_predictions(x)
        if aggregate == "mean":
            return m.mean(axis=0)
        if aggregate == "median":
            return np.median(m, axis=0)
        raise ValueError("aggregate must be 'mean' or 'median'")

    def quantiles(
        self, x: np.ndarray, qs: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    ) -> np.ndarray:
        return np.quantile(self.member_predictions(x), qs, axis=0)

    def inclusion_probabilities(self) -> dict[str, float]:
        """P(family selected | member), the structural-uncertainty readout.

        Normalizing by the number of members in which the family was OFFERED (it was
        in the member's random subset) rather than by all members, so a family is not
        penalized for having been absent from a member's menu.
        """
        offered: dict[str, int] = {f.key: 0 for f in self.families}
        selected: dict[str, int] = {f.key: 0 for f in self.families}
        for m in self.members:
            for key in m.subset_keys:
                offered[key] += 1
            selected[m.family.key] += 1
        return {
            k: (selected[k] / offered[k]) if offered[k] > 0 else 0.0 for k in offered
        }

    def selection_shares(self) -> dict[str, float]:
        """Fraction of members won by each family (sums to 1)."""
        shares: dict[str, int] = {f.key: 0 for f in self.families}
        for m in self.members:
            shares[m.family.key] += 1
        total = max(self.kept, 1)
        return {k: v / total for k, v in shares.items()}


def bape_fit(
    families: tuple[ModelFamily, ...],
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_members: int = 200,
    subset_size: int | None = None,
    seed: int = 0,
    block: int | None = None,
    n_starts: int = 6,
    criterion: str = "aicc",
) -> BapeEnsemble:
    """Fit a BAPE ensemble over the bank.

    subset_size defaults to ceil(sqrt(len(families))) + 1, the random-forest
    heuristic adapted to small libraries (guarantees >= 2 families per member when
    the bank has >= 2).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    n_fam = len(families)
    if n_fam == 0:
        raise ValueError("empty family bank")
    if subset_size is None:
        subset_size = min(n_fam, int(np.ceil(np.sqrt(n_fam))) + 1)
    subset_size = max(1, min(subset_size, n_fam))

    rng = np.random.default_rng(seed)
    idx = bootstrap_indices(n, n_members, rng, block=block)

    members: list[BapeMember] = []
    for b in range(n_members):
        xb, yb = x[idx[b]], y[idx[b]]
        chosen = rng.choice(n_fam, size=subset_size, replace=False)
        subset = tuple(families[int(c)] for c in chosen)
        fits: list[tuple[ModelFamily, FitResult]] = []
        for j, fam in enumerate(subset):
            res = fit_family(
                fam, xb, yb, n_starts=n_starts, seed=seed + 104729 * (b + 1) + j
            )
            if res.success and np.isfinite(res.rss):
                fits.append((fam, res))
        # coherent per-member criterion: preferred unless it is +inf for every
        # candidate in this member's menu (tiny resamples), then BIC for all
        crit = criterion
        if fits and not any(np.isfinite(getattr(r, criterion)) for _, r in fits):
            crit = "bic"
        best: tuple[ModelFamily, FitResult] | None = None
        best_val = float("inf")
        for fam, res in fits:
            val = getattr(res, crit)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best = (fam, res)
        if best is not None:
            members.append(
                BapeMember(
                    family=best[0],
                    fit=best[1],
                    boot_index=b,
                    subset_keys=tuple(f.key for f in subset),
                )
            )

    return BapeEnsemble(
        families=families,
        members=members,
        requested=n_members,
        meta={
            "subset_size": subset_size,
            "criterion": criterion,
            "seed": seed,
            "block": block,
            "n_starts": n_starts,
        },
    )
