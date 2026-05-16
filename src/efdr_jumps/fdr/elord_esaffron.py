from __future__ import annotations

from collections.abc import Iterable, Iterator


def elord(
    e_values: Iterable[float],
    alpha: float,
    w1: float = 0.05,
    phi: float = 0.5,
    psi: float = 0.5,
) -> Iterator[bool]:
    """e-LORD online FDR procedure (Zhang, Wei, Ren & Zou 2025 — e-GAI, Algorithm 1).

    At each step t (1-indexed), using RAI (Risk Aversion Investing):
        alpha_t = omega_t * rw_t * (R_{t-1} + 1)
    where rw_t = alpha - sum_{j<t} alpha_j / (R_{j-1}+1) is the remaining
    wealth, and omega_t is updated via equation (9) of the paper:
        omega_{t+1} = omega_t + w1 * phi^{t-R_t} * (1-delta_t)
                               - w1 * psi^{R_t}   * delta_t

    The recursive rw update is: rw_{t+1} = rw_t * (1 - omega_t).

    Reject H_t iff E_t >= 1 / alpha_t.  Controls FDR under arbitrary dependence.

    Parameters
    ----------
    e_values:
        Stream of e-values (one per hypothesis, in arrival order).
    alpha:
        Target FDR level in (0, 1).
    w1:
        Initial ω₁ parameter.  The paper recommends ω₁ = O(1/T); for streams
        of length T ≈ 500 use w1=0.005; for aggressive short streams use
        w1=0.2.  Larger w1 → more aggressive budget spending → more power but
        higher risk of early budget exhaustion.  Default 0.05.
    phi:
        Geometric growth factor for ω after non-rejections.  Paper default 0.5.
    psi:
        Geometric decay factor for ω after rejections.  Paper default 0.5.

    Yields
    ------
    bool
        True if H_t is rejected, False otherwise.
    """
    rw = alpha  # remaining wealth rw_1 = alpha
    omega = w1
    R = 0  # rejections before current step

    for t, e_t in enumerate(e_values, start=1):
        alpha_t = max(0.0, omega * rw * (R + 1))
        reject = bool(e_t >= 1.0 / alpha_t) if alpha_t > 0.0 else False
        delta_t = 1 if reject else 0
        R_new = R + delta_t

        # rw update uses omega before the update below (eq: rw_{t+1} = rw_t*(1-omega_t))
        rw = max(0.0, rw * (1.0 - omega))

        # omega update (eq. 9): adjust toward higher spending after non-rejections,
        # lower spending after rejections, to hedge against alpha-death
        omega = omega + w1 * phi ** (t - R_new) * (1 - delta_t) - w1 * psi**R_new * delta_t
        omega = max(0.0, min(1.0, omega))

        R = R_new
        yield reject


def esaffron(
    e_values: Iterable[float],
    alpha: float,
    w1: float = 0.05,
    phi: float = 0.5,
    psi: float = 0.5,
    lambda_cand: float = 0.1,
) -> Iterator[bool]:
    """e-SAFFRON online FDR procedure (Zhang, Wei, Ren & Zou 2025 — e-GAI, Algorithm 2).

    Adaptive variant of e-LORD: remaining wealth is only charged when a
    *non-candidate* hypothesis is tested.  A hypothesis is a candidate iff
    E_t >= 1/lambda_cand, i.e., the e-value already exceeds the candidacy bar.

    Formulae:
        rw_1 = alpha * (1 - lambda_cand)   (reduced initial wealth)
        alpha_t = omega_t * rw_t * (R_{t-1} + 1)    (same as e-LORD)
        delta_t = 1{E_t >= 1/alpha_t}               (reject regardless of candidacy)
        rw_{t+1} = rw_t * (1 - omega_t * 1{E_t < 1/lambda_cand})

    The wealth is saved when a strong candidate is observed (E_t >= 1/lambda_cand),
    because such hypotheses do not consume budget in the FDP estimator sum.
    This makes e-SAFFRON strictly more powerful than e-LORD in expectation.

    Rejection rule applies to ALL hypotheses (not only candidates).

    Parameters
    ----------
    e_values:
        Stream of e-values.
    alpha:
        Target FDR level in (0, 1).
    w1:
        Initial ω₁.  Paper default for T=500: 0.005.  Default here: 0.05.
    phi:
        Geometric growth factor after non-rejections.  Paper default 0.5.
    psi:
        Geometric decay factor after rejections.  Paper default 0.5.
    lambda_cand:
        Candidacy threshold in (0, 1).  A hypothesis with E_t >= 1/lambda_cand
        is a candidate.  Paper uses lambda_cand=0.1 (not 0.5 as in p-value
        SAFFRON).

    Yields
    ------
    bool
        True if H_t is rejected, False otherwise.
    """
    rw = alpha * (1.0 - lambda_cand)  # reduced initial wealth for SAFFRON
    omega = w1
    R = 0
    cand_threshold = 1.0 / lambda_cand  # E_t >= cand_threshold → candidate

    for t, e_t in enumerate(e_values, start=1):
        alpha_t = max(0.0, omega * rw * (R + 1))
        reject = bool(e_t >= 1.0 / alpha_t) if alpha_t > 0.0 else False
        delta_t = 1 if reject else 0
        R_new = R + delta_t

        # Only charge rw when E_t is a NON-candidate (e_t < 1/lambda)
        is_noncand = 1 if e_t < cand_threshold else 0
        rw = max(0.0, rw * (1.0 - omega * is_noncand))

        # omega update: same as e-LORD
        omega = omega + w1 * phi ** (t - R_new) * (1 - delta_t) - w1 * psi**R_new * delta_t
        omega = max(0.0, min(1.0, omega))

        R = R_new
        yield reject
