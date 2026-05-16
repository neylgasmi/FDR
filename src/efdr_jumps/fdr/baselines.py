from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import stats


def bh(p_values: Iterable[float], alpha: float) -> list[bool]:
    """Benjamini-Hochberg (1995) offline FDR procedure on p-values.

    Sort p-values ascending: p_(1) ≤ ... ≤ p_(n).
    k* = max{k : p_(k) ≤ alpha * k / n}, with k*=0 if no k qualifies.
    Reject all i with p_i ≤ alpha * k* / n.

    Controls FDR under independent or positive-regression-dependent (PRDS)
    p-values. Reference: Benjamini & Hochberg 1995, JRSS-B eq. (2).
    """
    pv = np.asarray(list(p_values), dtype=float)
    n = len(pv)
    if n == 0:
        return []

    order = np.argsort(pv)                       # ascending
    pv_sorted = pv[order]

    thresholds = alpha * np.arange(1, n + 1) / n   # alpha*k/n for k=1..n
    qualifying = np.where(pv_sorted <= thresholds)[0]

    if len(qualifying) == 0:
        return [False] * n

    k_star = int(qualifying[-1]) + 1
    cutoff = alpha * k_star / n

    return [bool(pv[i] <= cutoff) for i in range(n)]


def bh_lee_mykland(
    log_price: np.ndarray,
    alpha: float,
    dt_seconds: float,
    window: int = 100,
) -> list[bool]:
    """BH applied to Lee & Mykland (2008) two-sided normal p-values.

    L(i) = r_i / σ̂(i) is treated as N(0,1) under H0 (large-sample approx).
    Positions with NaN L(i) (insufficient history) are mapped to p=1 and
    never rejected.

    Reference: Yen (2013) §3.1.
    """
    from ..estimators.bipower import lee_mykland_stat

    L = lee_mykland_stat(log_price, dt_seconds, window=window)
    # Two-sided p-value: 2 * P(Z > |L(i)|)
    pv = np.where(np.isnan(L), 1.0, 2.0 * stats.norm.sf(np.abs(L)))
    return bh(pv, alpha)


def bh_bns(
    log_price: np.ndarray,
    alpha: float,
    dt_seconds: float,
    window: int = 100,
) -> list[bool]:
    """BH on per-return BNS-normalised Z-scores (per-return variant for BH-BNS).

    For each return i, estimate the local spot variance from bipower variation
    on the causal window r[max(0, i-window):i] (excludes r[i] itself), compute:
        Z_i = r_i / sqrt(bv_i * dt / SECS_PER_YEAR)
    then apply BH to the two-sided normal p-values.

    Using BV (Barndorff-Nielsen & Shephard 2004) for normalisation follows the
    spirit of Bajgrowicz-Scaillet (2016): jump-robust spot-variance estimation.
    Causal exclusion of r[i] prevents contamination of the null distribution.
    Positions where the window has fewer than 4 returns are mapped to p=1.
    """
    from ..estimators.bipower import bipower_variation
    from ..simulate.base import SECS_PER_YEAR as _SY

    r = np.diff(log_price)
    n = len(r)
    pv = np.ones(n)

    for i in range(n):
        if i < window:     # full window required: partial window gives unstable BV
            continue
        start = max(0, i - window)
        # sub = log_price[start:i+1] → returns r[start:i] (excludes r[i])
        sub = log_price[start : i + 1]
        if len(sub) < 5:
            continue
        bv_annual = bipower_variation(sub, dt_seconds)
        if bv_annual <= 0.0:
            continue
        sigma_sq_tick = bv_annual * (dt_seconds / _SY)
        z = r[i] / float(np.sqrt(sigma_sq_tick))
        pv[i] = 2.0 * float(stats.norm.sf(abs(z)))

    return bh(pv, alpha)


def bh_bns_global(
    log_price: np.ndarray,
    alpha: float,
    dt_seconds: float,
) -> bool:
    """BNS ratio test applied to the full path: returns a single boolean.

    Returns True if at least one jump is detected anywhere in the path.
    Use for day-level filtering, not per-return comparison.
    Reference: Bajgrowicz-Scaillet (2016).
    """
    from ..estimators.bipower import bns_ratio_stat
    from scipy import stats as _stats

    z = bns_ratio_stat(log_price, dt_seconds)
    return float(_stats.norm.sf(z)) <= alpha
