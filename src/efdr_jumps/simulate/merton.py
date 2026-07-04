from __future__ import annotations

import numpy as np

from .base import SECS_PER_YEAR, PathSimulator, SimulationResult
from .heston import HestonSimulator


class MertonSimulator(PathSimulator):
    """
    Heston stochastic volatility + compound Poisson-Merton jumps.

    The log-price dynamics are:
        d log S = (μ - V/2 - λ·k̄) dt + √V dW_S + J dN
    where J ~ N(μ_J, σ_J²) and k̄ = E[e^J] - 1 = exp(μ_J + σ_J²/2) - 1.

    Reference: Andersen-Benzoni-Lund (2002) parameterization.

    Parameters
    ----------
    heston  : underlying HestonSimulator (provides diffusion + variance)
    lam     : Poisson arrival rate of jumps (jumps per year)
    mu_j    : mean log-jump size
    sigma_j : std of log-jump size
    """

    def __init__(
        self,
        heston: HestonSimulator | None = None,
        lam: float = 5.0,
        mu_j: float = -0.01,
        sigma_j: float = 0.02,
    ) -> None:
        self.heston = heston if heston is not None else HestonSimulator()
        self.lam = lam
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    def simulate(
        self,
        n_steps: int,
        dt_seconds: float,
        rng: np.random.Generator,
    ) -> SimulationResult:
        dt = dt_seconds / SECS_PER_YEAR
        k_bar = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0

        # Heston with drift adjusted for jump mean-correction
        heston_adj = HestonSimulator(
            kappa=self.heston.kappa,
            theta=self.heston.theta,
            xi=self.heston.xi,
            rho=self.heston.rho,
            v0=self.heston.v0,
            mu=self.heston.mu - self.lam * k_bar,
        )
        base = heston_adj.simulate(n_steps, dt_seconds, rng)
        log_s = base.log_price.copy()

        # compound Poisson overlay
        lam_dt = self.lam * dt
        n_jumps_per_step = rng.poisson(lam_dt, n_steps)

        step_jump_sizes = np.zeros(n_steps + 1)
        jump_indices: list[int] = []

        for idx in np.where(n_jumps_per_step > 0)[0]:
            nj = int(n_jumps_per_step[idx])
            sizes = rng.normal(self.mu_j, self.sigma_j, nj)
            step_jump_sizes[idx + 1] = float(sizes.sum())
            jump_indices.append(int(idx + 1))

        # Propagate jumps forward cumulatively
        log_s += np.cumsum(step_jump_sizes)

        return SimulationResult(
            times=base.times,
            log_price=log_s,
            jump_indices=np.array(jump_indices, dtype=np.int64),
            sigma_path=base.sigma_path,
        )
