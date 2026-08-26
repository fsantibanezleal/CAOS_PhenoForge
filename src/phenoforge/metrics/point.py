"""Point-prediction metrics."""

from __future__ import annotations

import numpy as np


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y - yhat) ** 2)) / denom
