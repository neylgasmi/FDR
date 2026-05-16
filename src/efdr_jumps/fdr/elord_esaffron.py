from __future__ import annotations

from collections.abc import Iterable, Iterator

from .elond import _gamma


def elord(
    e_values: Iterable[float],
    alpha: float,
    w0: float = 0.5,
) -> Iterator[bool]:
    """e-LORD online FDR procedure (Zhang, Wei, Ren, Zou 2025 — e-GAI framework).

    At each step t (1-indexed):
        alpha_t = alpha * (w0 * gamma_t + (1-w0) * sum_{j: tau_j < t} gamma_{t - tau_j})
        Reject H_t  iff  E_t >= 1 / alpha_t

    where tau_j is the time of the j-th rejection and gamma_t is the
    Javanmard-Montanari sequence from elond.py.

    The initial wealth W_0 = w0 * alpha is invested via the gamma sequence
    from the start; each rejection at tau_j donates (1-w0)*alpha gamma-units
    for future steps.  Choosing w0=0.5 splits wealth evenly between the
    initial deposit and accumulated rejections.

    Parameters
    ----------
    e_values:
        Stream of e-values (one per hypothesis, in arrival order).
    alpha:
        Target FDR level in (0, 1).
    w0:
        Initial wealth fraction in (0, 1).  Default 0.5.

    Yields
    ------
    bool
        True if H_t is rejected, False otherwise.
    """
    rejection_times: list[int] = []   # tau_j for each rejection j

    for t, e_t in enumerate(e_values, start=1):
        # Contribution from initial wealth
        init_term = w0 * _gamma(t)

        # Contribution from past rejections
        past_term = sum(_gamma(t - tau) for tau in rejection_times if tau < t)

        alpha_t = alpha * (init_term + (1.0 - w0) * past_term)
        reject = e_t >= 1.0 / alpha_t if alpha_t > 0 else False
        if reject:
            rejection_times.append(t)
        yield reject
