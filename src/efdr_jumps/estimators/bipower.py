from __future__ import annotations

import numpy as np
from scipy.special import gamma

from ..simulate.base import SECS_PER_YEAR

# E[|Z|^p] constants for Z ~ N(0,1)
_MU1: float = np.sqrt(2.0 / np.pi)  # E[|Z|]
_MU43: float = 2.0 ** (2.0 / 3.0) * gamma(7.0 / 6.0) / gamma(0.5)  # E[|Z|^{4/3}]
# Asymptotic variance constant (Barndorff-Nielsen & Shephard 2006)
_BNS_THETA: float = (np.pi / 2.0) ** 2 + np.pi - 5.0


def realized_variance(log_price: np.ndarray, dt_seconds: float) -> float:
    """Annualized realized variance RV_n = Σ rᵢ² / T."""
    r = np.diff(log_price)
    T = len(r) * dt_seconds / SECS_PER_YEAR
    return float(np.sum(r**2) / T)


def bipower_variation(log_price: np.ndarray, dt_seconds: float) -> float:
    """
    Annualized bipower variation BV_n (Barndorff-Nielsen & Shephard 2004).
    Robust to isolated jumps in the limit Δ→0.
    Scale factor (π/2)·μ₁⁻² = π/2 since μ₁² = 2/π.
    """
    r = np.diff(log_price)
    n = len(r)
    bv_raw = (np.pi / 2.0) * (n / (n - 1)) * np.sum(np.abs(r[:-1]) * np.abs(r[1:]))
    T = n * dt_seconds / SECS_PER_YEAR
    return float(bv_raw / T)


def bns_ratio_stat(log_price: np.ndarray, dt_seconds: float) -> float:
    """
    BNS ratio jump test statistic (Barndorff-Nielsen & Shephard 2006).
    Large positive values indicate jumps.
    Reference: Huang & Tauchen (2005) eq. (8).
    """
    r = np.diff(log_price)
    n = len(r)
    rv = float(np.sum(r**2))
    bv = (np.pi / 2.0) * float(np.sum(np.abs(r[:-1]) * np.abs(r[1:])))

    if rv == 0.0 or bv == 0.0:
        return 0.0

    # Tri-power quarticity (ADS 2012 eq. 5), used in variance estimation
    tpq = (
        _MU43 ** (-3)
        * n
        / (n - 2)
        * float(
            np.sum(
                np.abs(r[:-2]) ** (4.0 / 3.0)
                * np.abs(r[1:-1]) ** (4.0 / 3.0)
                * np.abs(r[2:]) ** (4.0 / 3.0)
            )
        )
    )

    rel_jump = (rv - bv) / rv
    denom = np.sqrt(_BNS_THETA * max(1.0, tpq / bv**2) / n)
    return float(rel_jump / denom) if denom > 0 else 0.0


def lee_mykland_stat(
    log_price: np.ndarray,
    dt_seconds: float,
    window: int = 100,
) -> np.ndarray:
    """
    Lee & Mykland (2008) local jump statistic L(i) = r_i / σ̂(i).

    σ̂(i) is estimated from bipower variation over the `window` returns
    strictly before return i — the test point is never included in its own
    estimator window, preserving e-value validity.

    Returns NaN for i < window.
    """
    r = np.diff(log_price)
    n = len(r)
    K = window

    # P[j] = |r[j]| × |r[j+1]|  — consecutive absolute-return products
    P = np.abs(r[:-1]) * np.abs(r[1:])  # shape (n-1,)
    cumP = np.empty(n, dtype=np.float64)
    cumP[0] = 0.0
    np.cumsum(P, out=cumP[1:])  # cumP[j] = Σ P[0:j]

    L = np.full(n, np.nan)
    if K >= n:
        return L

    # For i in [K, n-1]: window is r[i-K:i], giving K-1 consecutive products
    # P[i-K], …, P[i-2]  →  sum = cumP[i-1] - cumP[i-K]
    i_range = np.arange(K, n)
    bv_sums = cumP[i_range - 1] - cumP[i_range - K]
    # Estimated σ²·Δt from (π/2) × (1/(K-1)) × Σ products
    sigma_sq_dt = (np.pi / 2.0) / (K - 1) * bv_sums
    valid = sigma_sq_dt > 0.0
    L[i_range[valid]] = r[i_range[valid]] / np.sqrt(sigma_sq_dt[valid])
    return L
