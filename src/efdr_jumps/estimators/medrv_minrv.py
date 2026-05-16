from __future__ import annotations

import numpy as np

from ..simulate.base import SECS_PER_YEAR

# Scale factors from Andersen, Dobrev, Schaumburg (2012) eqs. (12)–(13).
# They correct for E[min(|Z₁|,|Z₂|)²] = (π-2)/π and
# E[med(|Z₁|,|Z₂|,|Z₃|)²] = (6-4√3+π)/π.
_SCALE_MIN: float = np.pi / (np.pi - 2.0)               # ≈ 2.7141
_SCALE_MED: float = np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi)  # ≈ 1.4191


def minrv(log_price: np.ndarray, dt_seconds: float) -> float:
    """
    MinRV annualized estimator (Andersen, Dobrev, Schaumburg 2012, eq. 13).
    More robust to jumps than BV: min(|r_{i-1}|,|r_i|) ≈ non-jump return.
    """
    r = np.diff(log_price)
    n = len(r)
    iv_raw = _SCALE_MIN * n / (n - 1) * float(
        np.sum(np.minimum(np.abs(r[:-1]), np.abs(r[1:])) ** 2)
    )
    T = n * dt_seconds / SECS_PER_YEAR
    return iv_raw / T


def medrv(log_price: np.ndarray, dt_seconds: float) -> float:
    """
    MedRV annualized estimator (Andersen, Dobrev, Schaumburg 2012, eq. 12).
    Median of three consecutive absolute returns zeroes out isolated jumps.
    Requires at least 3 returns (4 prices).
    """
    r = np.diff(log_price)
    n = len(r)
    if n < 3:
        return float("nan")
    ar = np.abs(r)
    # Stack |r_{i-1}|, |r_i|, |r_{i+1}| as rows, take per-row median
    triplets = np.stack([ar[:-2], ar[1:-1], ar[2:]], axis=1)
    meds = np.median(triplets, axis=1)
    iv_raw = _SCALE_MED * n / (n - 2) * float(np.sum(meds**2))
    T = n * dt_seconds / SECS_PER_YEAR
    return iv_raw / T
