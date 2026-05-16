from __future__ import annotations

from collections.abc import Callable

import numpy as np


def spot_variance(
    log_price: np.ndarray,
    dt_seconds: float,
    estimator_fn: Callable[[np.ndarray, float], float],
    K: int = 100,
) -> np.ndarray:
    """
    Annualized spot variance σ̂²(i) for each return index i ∈ {0, …, n-1}.

    The window used for σ̂²(i) is the K log-price observations strictly before
    index i+1, i.e., log_price[max(0, i-K+1) : i+1], which corresponds to
    returns r[max(0, i-K) : i].  Return r[i] is NEVER included in its own
    estimator window — this is required for e-value validity (see CLAUDE.md §3).

    Parameters
    ----------
    log_price    : (N+1,) array of log-prices
    dt_seconds   : sampling interval in seconds
    estimator_fn : function(sub_log_price, dt_seconds) → annualized IV (float)
    K            : backward window length in observations (number of returns)

    Returns
    -------
    sigma_sq : (N,) array, NaN where the window has fewer than 2 returns.
    """
    n = len(log_price) - 1  # number of returns
    sigma_sq = np.full(n, np.nan)

    for i in range(n):
        # Window: r[max(0, i-K) : i]  →  log_price[max(0, i-K) : i+1]
        start = max(0, i - K)
        sub = log_price[start : i + 1]  # i+1-start prices → i-start returns
        if len(sub) < 4:  # need ≥ 3 returns (medrv needs triplets)
            continue
        sigma_sq[i] = estimator_fn(sub, dt_seconds)

    return sigma_sq


def assert_window_excludes_test_point(
    log_price: np.ndarray,
    dt_seconds: float,
    estimator_fn: Callable[[np.ndarray, float], float],
    K: int,
    i: int,
) -> None:
    """
    Assertion helper: verify that the window for return i does NOT include r[i].

    Computes σ̂²(i) twice — once normally, once after zeroing out r[i] in a
    copy.  If the window is correct, the two estimates must agree exactly.
    """
    start = max(0, i - K)

    # Normal estimate
    sub_normal = log_price[start : i + 1]
    est_normal = estimator_fn(sub_normal, dt_seconds)

    # Estimate after corrupting r[i] = log_price[i+1] - log_price[i]
    lp_corrupt = log_price.copy()
    # shift log_price[i+1:] by a large value so r[i] is huge
    lp_corrupt[i + 1 :] += 1e6
    sub_corrupt = lp_corrupt[start : i + 1]  # unchanged: stops at i+1 (exclusive)
    est_corrupt = estimator_fn(sub_corrupt, dt_seconds)

    assert est_normal == est_corrupt, (
        f"Window for i={i} includes the test point: "
        f"est_normal={est_normal:.6e}, est_corrupt={est_corrupt:.6e}"
    )
