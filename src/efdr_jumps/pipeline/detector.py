from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..simulate.base import SECS_PER_YEAR

# jump rate (jumps/year) per regime for n_steps=500, dt=5s.
# expected jumps per path = lam * n_steps * dt / SECS_PER_YEAR
#   rare: ~0.64   moderate: ~2.12   dense: ~6.37
_REGIME_LAM: dict[str, float] = {
    "heston_h0": 0.0,  # pure diffusion, no jumps
    "rare": 1_500.0,
    "moderate": 5_000.0,
    "dense": 15_000.0,
    "hawkes_dense": 15_000.0,  # aliases "dense" for the Merton approximation
}

_ESTIMATOR_FNS: dict[str, object] = {}  # populated lazily on first import


def _get_estimator_fn(kind: str):
    global _ESTIMATOR_FNS
    if not _ESTIMATOR_FNS:
        from ..estimators import bipower_variation, medrv, minrv, threshold_rv

        _ESTIMATOR_FNS = {
            "medrv": medrv,
            "minrv": minrv,
            "bv": bipower_variation,
            "threshold": threshold_rv,
        }
    if kind not in _ESTIMATOR_FNS:
        raise ValueError(f"Unknown estimator '{kind}'. Choose from {list(_ESTIMATOR_FNS)}")
    return _ESTIMATOR_FNS[kind]


# config dataclasses
@dataclass
class SimulatorConfig:
    """Simulation parameters for one run."""

    regime: str = "moderate"  # "heston_h0", "rare", "moderate", "dense"
    dt_seconds: float = 5.0
    n_steps: int = 500
    # Heston SV params
    kappa: float = 2.0
    theta: float = 0.04  # long-run variance → σ_base ≈ 0.20 annualized
    xi: float = 0.5
    rho: float = -0.7
    v0: float = 0.04
    # Jump size (in multiples of the local vol * sqrt(dt))
    jump_size_sigma: float = 5.0
    # Spread of jump size distribution as fraction of |mu_j|
    jump_sigma_spread: float = 0.1


@dataclass
class EstimatorConfig:
    """Spot-variance estimator parameters."""

    kind: str = "medrv"  # "medrv", "minrv", "bv", "threshold"
    window: int = 50  # backward window in number of returns


@dataclass
class FDRConfig:
    """FDR algorithm parameters."""

    algo: str = "elond"
    alpha: float = 0.05
    # e-LORD / e-SAFFRON params
    w1: float = 0.05
    phi: float = 0.5
    psi: float = 0.5
    lambda_cand: float = 0.1
    # BH-LM params
    lm_window: int = 100


@dataclass
class DetectorResult:
    """Output of a single run_detector call."""

    rejection_set: frozenset[int]  # 0-indexed return indices
    true_jumps: frozenset[int]  # 0-indexed return indices
    n_hypotheses: int
    wall_clock: float  # seconds for the full run
    sim_config: SimulatorConfig = field(repr=False)
    est_config: EstimatorConfig = field(repr=False)
    fdr_config: FDRConfig = field(repr=False)
    seed: int = 0


# core orchestrator
def run_detector(
    sim_config: SimulatorConfig,
    est_config: EstimatorConfig,
    fdr_config: FDRConfig,
    seed: int = 0,
) -> DetectorResult:
    """Simulate → estimate vol → build e-values → run FDR → return result.

    The spot-variance window excludes the test tick, which is required for
    e-value validity (see estimators/spot.py).

    Jump indices from PathSimulator are log-price indices (1-indexed), so
    the corresponding return index is j-1 for a jump at log_price[j].
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # 1. simulate
    from ..simulate import HestonSimulator, MertonSimulator

    heston = HestonSimulator(
        kappa=sim_config.kappa,
        theta=sim_config.theta,
        xi=sim_config.xi,
        rho=sim_config.rho,
        v0=sim_config.v0,
    )

    lam = _REGIME_LAM[sim_config.regime]
    if lam > 0.0:
        sigma_base = float(np.sqrt(sim_config.theta))
        dt_years = sim_config.dt_seconds / SECS_PER_YEAR
        mu_j = sim_config.jump_size_sigma * sigma_base * float(np.sqrt(dt_years))
        sigma_j = sim_config.jump_sigma_spread * abs(mu_j)
        simulator = MertonSimulator(heston=heston, lam=lam, mu_j=mu_j, sigma_j=sigma_j)
    else:
        simulator = heston

    result = simulator.simulate(sim_config.n_steps, sim_config.dt_seconds, rng)
    log_price = result.log_price
    r = np.diff(log_price)  # (n_steps,) returns
    n = len(r)

    # ground truth: convert log_price jump indices to return indices (0-based)
    true_jumps = frozenset(int(j) - 1 for j in result.jump_indices if 1 <= int(j) <= n)

    # 2. dispatch to bh baselines or e-value procedures
    algo = fdr_config.algo
    alpha = fdr_config.alpha

    if algo == "bh_lm":
        from ..fdr import bh_lee_mykland

        rejects_list = bh_lee_mykland(
            log_price, alpha, sim_config.dt_seconds, window=fdr_config.lm_window
        )
        rejection_set = frozenset(i for i, v in enumerate(rejects_list) if v)

    elif algo == "bh_bns":
        from ..fdr import bh_bns_global

        # day-level global test: same decision applied to every return
        detected = bh_bns_global(log_price, alpha, sim_config.dt_seconds)
        r = np.diff(log_price)
        rejects_list = [detected] * len(r)
        rejection_set = frozenset(i for i, v in enumerate(rejects_list) if v)
    else:
        # e-value based procedures
        e_values = _compute_evalues(log_price, sim_config, est_config)
        rejection_set = _run_evalue_algo(e_values, alpha, fdr_config)

    wall_clock = time.perf_counter() - t0
    return DetectorResult(
        rejection_set=rejection_set,
        true_jumps=true_jumps,
        n_hypotheses=n,
        wall_clock=wall_clock,
        sim_config=sim_config,
        est_config=est_config,
        fdr_config=fdr_config,
        seed=seed,
    )


# internal helpers
def _compute_evalues(
    log_price: np.ndarray,
    sim_config: SimulatorConfig,
    est_config: EstimatorConfig,
) -> np.ndarray:
    """Compute e-values for each return using windowed spot vol estimate."""
    from ..estimators import spot_variance
    from ..evalues import construct_evalue

    estimator_fn = _get_estimator_fn(est_config.kind)
    dt_years = sim_config.dt_seconds / SECS_PER_YEAR
    r = np.diff(log_price)
    n = len(r)

    sigma_sq = spot_variance(log_price, sim_config.dt_seconds, estimator_fn, K=est_config.window)

    e_values = np.ones(n)
    for i in range(n):
        sv = sigma_sq[i]
        if np.isnan(sv) or sv <= 0.0:
            continue
        # Skip until the window is fully filled: a partial window gives an
        # unreliable (often severely downward-biased) sigma estimate, which
        # makes E_i -> inf even under H0, violating e-value validity.
        if i < est_config.window:
            continue
        e_values[i] = construct_evalue(
            r_i=float(r[i]),
            sigma_hat_i=float(np.sqrt(sv)),
            dt_years=dt_years,
        )
    return e_values


def _run_evalue_algo(
    e_values: np.ndarray,
    alpha: float,
    fdr_config: FDRConfig,
) -> frozenset[int]:
    """Dispatch e-values to the requested FDR algorithm."""
    from ..fdr import ebh, elond, elord, esaffron, stopped_ebh

    algo = fdr_config.algo

    if algo == "ebh":
        rejects = ebh(e_values, alpha)
    elif algo == "elond":
        rejects = list(elond(e_values, alpha))
    elif algo == "elord":
        rejects = list(
            elord(e_values, alpha, w1=fdr_config.w1, phi=fdr_config.phi, psi=fdr_config.psi)
        )
    elif algo == "esaffron":
        rejects = list(
            esaffron(
                e_values,
                alpha,
                w1=fdr_config.w1,
                phi=fdr_config.phi,
                psi=fdr_config.psi,
                lambda_cand=fdr_config.lambda_cand,
            )
        )
    elif algo == "stopped_ebh":
        rejects = list(stopped_ebh(e_values, alpha))
    else:
        raise ValueError(f"Unknown e-value algo '{algo}'.")

    return frozenset(i for i, v in enumerate(rejects) if v)
