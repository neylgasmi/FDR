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
) -> list[bool]:
    """BH applied to the BNS ratio statistic treated as a single global test.

    The BNS statistic (Barndorff-Nielsen & Shephard 2006) tests the whole path
    for jump presence. Here we return a path-level boolean: True for every
    observation if the global H0 is rejected at level alpha, False otherwise.
    This matches the way Bajgrowicz-Scaillet (2016) use BNS as a day-level filter.
    """
    from ..estimators.bipower import bns_ratio_stat

    z = bns_ratio_stat(log_price, dt_seconds)
    p_global = float(stats.norm.sf(z))    # one-sided: large z → jump present
    n = len(log_price) - 1
    rejected_globally = p_global <= alpha
    return [rejected_globally] * n
