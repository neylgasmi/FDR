from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

# normalization constant for the Javanmard-Montanari gamma sequence.
# gamma_t = _C_JM * log(max(t,2)) / (t * exp(sqrt(log(max(t,2)))))
# calibrated so that sum_t gamma_t = 1 (verified by truncated summation to
# t=10^6, see docs/DECISIONS.md for the derivation)
_C_JM: float = 0.15708906


def _gamma(t: int) -> float:
    """Javanmard-Montanari canonical gamma_t, t ≥ 1."""
    lt = np.log(max(t, 2))
    return _C_JM * lt / (t * np.exp(np.sqrt(lt)))


def elond(
    e_values: Iterable[float],
    alpha: float,
) -> Iterator[bool]:
    """e-LOND online FDR procedure (Xu & Ramdas 2024, Algorithm 1).

    At each step t (1-indexed):
        alpha_t = alpha * gamma_t * (R_{t-1} + 1)
        Reject H_t  iff  E_t >= 1 / alpha_t

    where R_{t-1} is the number of rejections before step t.
    Controls FDR under arbitrary dependence.

    Parameters
    ----------
    e_values:
        Stream of e-values (one per hypothesis, in arrival order).
    alpha:
        Target FDR level in (0, 1).

    Yields
    ------
    bool
        True if H_t is rejected, False otherwise.
    """
    R = 0  # rejections so far
    for t, e_t in enumerate(e_values, start=1):
        alpha_t = alpha * _gamma(t) * (R + 1)
        reject = e_t >= 1.0 / alpha_t if alpha_t > 0 else False
        if reject:
            R += 1
        yield reject
