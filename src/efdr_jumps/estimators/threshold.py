from __future__ import annotations

import numpy as np

from ..simulate.base import SECS_PER_YEAR
from .bipower import bipower_variation


def threshold_rv(
    log_price: np.ndarray,
    dt_seconds: float,
    *,
    threshold_factor: float = 3.0,
    beta: float = 0.49,
    sigma_rough: float | None = None,
) -> float:
    """
    Truncated realized variance (Mancini 2009).

    Returns Σ rᵢ² · 1(|rᵢ| ≤ uₙ) / T  (annualized).

    Threshold: uₙ = threshold_factor · σ̂_rough · Δᵝ (Mancini & Renò 2011).
    β=0.49 places uₙ just below the Δ^{1/2} diffusive scale, so that any
    increment larger than uₙ is attributed to a jump with high probability.

    Parameters
    ----------
    threshold_factor : multiplier c in uₙ = c · σ̂ · Δᵝ  (default 3)
    beta             : exponent β ∈ (0, 1/2)  (default 0.49)
    sigma_rough      : preliminary annualized vol estimate; if None, estimated
                       from BV over the full path.
    """
    r = np.diff(log_price)
    n = len(r)
    dt = dt_seconds / SECS_PER_YEAR

    if sigma_rough is None:
        sigma_rough = np.sqrt(max(bipower_variation(log_price, dt_seconds), 1e-12))

    u = threshold_factor * sigma_rough * dt**beta
    iv_raw = float(np.sum(r**2 * (np.abs(r) <= u)))
    T = n * dt
    return iv_raw / T
