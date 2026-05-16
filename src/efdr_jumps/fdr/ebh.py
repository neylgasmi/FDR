from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def ebh(e_values: Iterable[float], alpha: float) -> list[bool]:
    """Offline e-BH procedure (Wang & Ramdas 2022, Algorithm 1).

    Sort e-values descending: E_(1) ≥ E_(2) ≥ ... ≥ E_(n).
    k* = max{k : E_(k) ≥ n / (alpha * k)}, with k*=0 if no k qualifies.
    Reject all i with E_i ≥ n / (alpha * k*).

    Parameters
    ----------
    e_values:
        Finite sequence of e-values (consumes the full stream).
    alpha:
        FDR level in (0, 1).

    Returns
    -------
    list[bool]
        Rejection decisions in input order (True = reject).
    """
    ev = np.asarray(list(e_values), dtype=float)
    n = len(ev)
    if n == 0:
        return []

    order = np.argsort(ev)[::-1]  # indices that sort ev descending
    ev_sorted = ev[order]  # E_(1) ≥ ... ≥ E_(n)

    thresholds = n / (alpha * np.arange(1, n + 1))  # n/(alpha*k) for k=1..n
    qualifying = np.where(ev_sorted >= thresholds)[0]

    if len(qualifying) == 0:
        return [False] * n

    k_star = int(qualifying[-1]) + 1  # largest k such that E_(k) >= n/(alpha*k)
    cutoff = n / (alpha * k_star)

    return [bool(ev[i] >= cutoff) for i in range(n)]
