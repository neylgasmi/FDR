from __future__ import annotations

import numpy as np
import pytest

from efdr_jumps.fdr.baselines import bh
from efdr_jumps.fdr.ebh import ebh

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_HESTON = None
_MERTON = None


def _get_simulators():
    global _HESTON, _MERTON
    if _HESTON is None:
        from efdr_jumps.simulate import HestonSimulator, MertonSimulator

        _HESTON = HestonSimulator(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
        _MERTON = MertonSimulator(heston=_HESTON, lam=5.0, mu_j=-0.01, sigma_j=0.02)
    return _HESTON, _MERTON


def _pareto_evalues(n: int, rng: np.random.Generator) -> np.ndarray:
    """Independent calibrated Pareto-1 e-values: E = U^{-1}, U~Uniform(0,1).

    These satisfy E[E] = 1 exactly under H0 and are independent.
    """
    u = rng.uniform(0.0, 1.0, size=n)
    return 1.0 / u


def _heston_evalues_h0(n: int, dt_s: float, rng: np.random.Generator) -> np.ndarray:
    """E-values from pure-diffusion Heston path (no jumps)."""
    from efdr_jumps.estimators import medrv
    from efdr_jumps.evalues import construct_evalue
    from efdr_jumps.simulate import SECS_PER_YEAR

    heston, _ = _get_simulators()
    res = heston.simulate(n + 1, dt_s, rng)
    r = np.diff(res.log_price)
    sigma_hat = np.sqrt(medrv(res.log_price, dt_s))
    dt_years = dt_s / SECS_PER_YEAR
    return np.array([construct_evalue(r[i], sigma_hat, dt_years) for i in range(n)])


# ---------------------------------------------------------------------------
# e-BH: deterministic stream
# ---------------------------------------------------------------------------


def test_ebh_deterministic():
    """E=(1,1,100,1,1), alpha=0.1 → reject only index 2 (third element)."""
    ev = [1.0, 1.0, 100.0, 1.0, 1.0]
    result = ebh(ev, alpha=0.1)
    # n=5, alpha=0.1.  E_(1)=100 >= 5/(0.1*1)=50 → k*=1, cutoff=50.
    # Only E_2=100 >= 50.
    assert result == [False, False, True, False, False], f"Got {result}"


def test_ebh_no_rejection():
    """All e-values=1, alpha=0.1 → no rejections (1 < 50)."""
    ev = [1.0] * 5
    result = ebh(ev, alpha=0.1)
    assert result == [False] * 5


def test_ebh_all_reject():
    """Large uniform e-values → all rejected."""
    n = 10
    ev = [200.0] * n
    result = ebh(ev, alpha=0.1)
    # E_(1)=200 >= n/(alpha*1) = 10/0.1=100 → k*=10 (all qualify), cutoff=100/10=10
    # All ev=200 >= 10
    assert all(result)


def test_ebh_empty():
    assert ebh([], alpha=0.1) == []


# ---------------------------------------------------------------------------
# e-BH: self-validity under H0 — Pareto stream
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ebh_fdr_validity_pareto():
    """FDR ≤ alpha + 2·SE on Pareto-1 e-value stream (M=1000 reps)."""
    M = 1000
    n = 1000
    alpha = 0.1
    rng = np.random.default_rng(42)
    rejections = []
    for _ in range(M):
        ev = _pareto_evalues(n, rng)
        rej = ebh(ev, alpha=alpha)
        r = sum(rej)
        # All e-values are H0, so every rejection is a false discovery.
        rejections.append(r / max(r, 1))  # FDP = R / max(R,1) since all H0

    fdr = np.mean(rejections)
    se = np.std(rejections) / np.sqrt(M)
    assert fdr <= alpha + 2 * se, f"e-BH Pareto FDR={fdr:.4f} > alpha+2SE={alpha + 2*se:.4f}"


# ---------------------------------------------------------------------------
# e-BH: self-validity under H0 — Heston diffusion stream
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ebh_fdr_validity_heston():
    """FDR ≤ alpha + 2·SE on Heston pure-diffusion e-values (M=1000 reps)."""
    M = 1000
    n = 500
    dt_s = 5.0
    alpha = 0.1
    rng = np.random.default_rng(7)
    fdps = []
    for _ in range(M):
        ev = _heston_evalues_h0(n, dt_s, rng)
        rej = ebh(ev, alpha=alpha)
        r = sum(rej)
        fdps.append(r / max(r, 1))

    fdr = np.mean(fdps)
    se = np.std(fdps) / np.sqrt(M)
    assert fdr <= alpha + 2 * se, f"e-BH Heston FDR={fdr:.4f} > alpha+2SE={alpha + 2*se:.4f}"


# ---------------------------------------------------------------------------
# BH baselines: deterministic stream
# ---------------------------------------------------------------------------


def test_bh_deterministic_reject_one():
    """p=(0.9, 0.9, 0.001, 0.9, 0.9), alpha=0.1 → reject only index 2."""
    pv = [0.9, 0.9, 0.001, 0.9, 0.9]
    result = bh(pv, alpha=0.1)
    # p_(1)=0.001 <= 0.1*1/5=0.02 → k*=1, cutoff=0.02. Only p[2]=0.001 <= 0.02.
    assert result == [False, False, True, False, False], f"Got {result}"


def test_bh_no_rejection():
    """Uniform p-values=0.5, alpha=0.1 → no rejections."""
    pv = [0.5] * 5
    result = bh(pv, alpha=0.1)
    assert result == [False] * 5


def test_bh_all_reject():
    """Very small p-values → all rejected."""
    pv = [1e-6] * 10
    result = bh(pv, alpha=0.1)
    assert all(result)


def test_bh_empty():
    assert bh([], alpha=0.1) == []


# ---------------------------------------------------------------------------
# BH baselines: self-validity under H0 — uniform p-values
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_bh_fdr_validity_uniform():
    """FDR ≤ alpha + 2·SE on Uniform(0,1) p-values (M=1000 reps)."""
    M = 1000
    n = 1000
    alpha = 0.1
    rng = np.random.default_rng(11)
    fdps = []
    for _ in range(M):
        pv = rng.uniform(0.0, 1.0, size=n)
        rej = bh(pv, alpha=alpha)
        r = sum(rej)
        fdps.append(r / max(r, 1))

    fdr = np.mean(fdps)
    se = np.std(fdps) / np.sqrt(M)
    assert fdr <= alpha + 2 * se, f"BH uniform FDR={fdr:.4f} > alpha+2SE={alpha + 2*se:.4f}"
