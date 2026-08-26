"""Core abstractions of the family bank.

A ModelFamily is a named, cited, bounded phenomenological model: a closed-form (or
numerically evaluated) response y = f(x; theta) with physically meaningful parameters,
declared units and bounds, and a declared calibration-data contract (what kind of data
can calibrate it). Families are the BASE LEARNERS of every ensemble method in
phenoforge; they are deliberately simple, transparent objects.

Design rules:
- Pure numpy in predict(); no global state; vectorized over x.
- Bounds are physical, not numerical conveniences; fits are always bounded.
- Every family carries its primary reference (authors, year, DOI where one exists).
  References are transcribed from the Fragua research dossier
  wip/fragua/research/mining-model-families-2026-08-25.md (CAOS_MANAGE).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class DataKind(str, Enum):
    """Calibration-data contract tags: what data a family needs.

    A dataset advertises the kinds it provides; the router matches them against
    each family's declared needs (see phenoforge.router).
    """

    TIMED_RECOVERY = "timed_recovery"          # batch flotation: (t, cumulative recovery)
    CONTINUOUS_RECOVERY = "continuous_recovery"  # plant: (residence time / flows, recovery)
    SIZE_ENERGY = "size_energy"                # comminution: (F80, P80, specific energy)
    PSD_TIME = "psd_time"                      # batch grinding: PSD vs time
    SETTLING_CURVE = "settling_curve"          # thickening: interface height vs time
    CONVERSION_TIME = "conversion_time"        # leaching: conversion vs time
    XY_RESPONSE = "xy_response"                # generic tabular response (features -> target)
    ANNUAL_BALANCE = "annual_balance"          # utility series: annual consumption balances


@dataclass(frozen=True)
class Param:
    """One physical parameter of a family."""

    name: str
    unit: str
    low: float
    high: float
    init: float
    meaning: str

    def __post_init__(self) -> None:
        if not (self.low < self.high):
            raise ValueError(f"{self.name}: low must be < high")
        if not (self.low <= self.init <= self.high):
            raise ValueError(f"{self.name}: init must lie within [low, high]")


@dataclass(frozen=True)
class Reference:
    """Primary reference for a family (real citation, never invented)."""

    text: str
    doi: str | None = None


@dataclass(frozen=True)
class ModelFamily:
    """A phenomenological model family: y = predict(x, theta).

    x is the independent variable array; for multi-input families x has shape
    (n, d) and the family documents its column order in `x_doc`.
    """

    key: str
    name: str
    process: str  # flotation | comminution | thickening | water | energy | leaching | generic
    params: tuple[Param, ...]
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    needs: tuple[DataKind, ...]
    equation: str  # LaTeX, for docs/UI; single source of truth mirrored by the web contract
    assumptions: str
    references: tuple[Reference, ...]
    x_doc: str = "x: 1-D independent variable"

    @property
    def k(self) -> int:
        """Number of free parameters."""
        return len(self.params)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([p.low for p in self.params], dtype=float)
        hi = np.array([p.high for p in self.params], dtype=float)
        return lo, hi

    @property
    def inits(self) -> np.ndarray:
        return np.array([p.init for p in self.params], dtype=float)

    def predict(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape[-1] != self.k:
            raise ValueError(f"{self.key}: theta has {theta.shape[-1]} values, expected {self.k}")
        return self.fn(np.asarray(x, dtype=float), theta)

    def residuals(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) - self.predict(x, theta)


@dataclass
class FitResult:
    """Outcome of one bounded fit of one family on one dataset (or resample)."""

    family_key: str
    theta: np.ndarray
    rss: float
    n: int
    success: bool
    n_starts: int
    seed: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return int(self.theta.shape[-1])

    @property
    def sigma2(self) -> float:
        """ML estimate of the gaussian error variance."""
        return max(self.rss / max(self.n, 1), 1e-300)

    @property
    def loglik(self) -> float:
        """Gaussian log-likelihood at the ML variance estimate."""
        n = self.n
        return -0.5 * n * (np.log(2.0 * np.pi * self.sigma2) + 1.0)

    @property
    def aic(self) -> float:
        """AIC with the variance counted as a fitted parameter (Burnham-Anderson 2004,
        DOI 10.1177/0049124104268644)."""
        p = self.k + 1
        return -2.0 * self.loglik + 2.0 * p

    @property
    def aicc(self) -> float:
        """Small-sample corrected AIC. Returns +inf when n <= p + 1 (the correction
        denominator vanishes), which is itself the honest verdict for an
        over-parameterized fit."""
        p = self.k + 1
        denom = self.n - p - 1
        if denom <= 0:
            return float("inf")
        return self.aic + (2.0 * p * (p + 1)) / denom

    @property
    def bic(self) -> float:
        p = self.k + 1
        return -2.0 * self.loglik + p * np.log(max(self.n, 1))
