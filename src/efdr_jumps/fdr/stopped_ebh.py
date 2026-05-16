from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np


def stopped_ebh(
    e_values: Iterable[float],
    alpha: float,
) -> Iterator[bool]:
    """Stopped e-BH (Wang, Dandapanthula & Ramdas 2025, arXiv:2502.08539).

    At each step t, recompute offline e-BH on E_1, ..., E_t and yield
    whether H_t is currently in the rejection set.

    Anytime-validity holds under the causal condition: each E_i is valid
    with respect to the filtration up to (but not including) observation i.
    In our pipeline this is guaranteed because the spot-volatility window
    excludes the test point i.

    Complexity: O(t log t) per step due to sort; O(T^2 log T) overall.
    For streaming use with T ≤ ~10^4 this is acceptable.

    Parameters
    ----------
    e_values:
        Stream of e-values (one per hypothesis, in arrival order).
    alpha:
        Target FDR level in (0, 1).

    Yields
    ------
    bool
        True if H_t is in the current e-BH rejection set at time t.
    """
    buffer: list[float] = []

    for e_t in e_values:
        buffer.append(e_t)
        t = len(buffer)
        ev = np.asarray(buffer, dtype=float)

        order = np.argsort(ev)[::-1]
        ev_sorted = ev[order]
        thresholds = t / (alpha * np.arange(1, t + 1))
        qualifying = np.where(ev_sorted >= thresholds)[0]

        if len(qualifying) == 0:
            yield False
        else:
            k_star = int(qualifying[-1]) + 1
            cutoff = t / (alpha * k_star)
            yield bool(e_t >= cutoff)
