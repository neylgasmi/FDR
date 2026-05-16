from __future__ import annotations

import numpy as np

from .base import SECS_PER_YEAR, PathSimulator, SimulationResult
from .heston import HestonSimulator


class HawkesSimulator(PathSimulator):
    """Heston diffusion + self-exciting (Hawkes) jump arrivals.

    Jump intensity follows the continuous-time Hawkes recursion
    discretised to ticks of size dt_seconds:

        λ_{n+1} = μ + (λ_n − μ)·exp(−β·Δt) + α·N_n

    where N_n ∈ {0,1} is the jump indicator at tick n.

    Branching ratio in continuous time: m = α / (β × SECS_PER_YEAR).
    For process stability, enforce m < 1 (checked at construction).

    Note: the discrete approximation is accurate for small Δt.  At large
    tick sizes (≥ 60s) the discretisation error grows; the process remains
    well-defined but clustering is less pronounced than in continuous time.

    Parameters
    ----------
    heston : HestonSimulator | None
        Underlying stochastic-volatility diffusion.
    mu_per_year : float
        Baseline Poisson arrival rate (jumps/year).
    alpha_per_year : float
        Per-jump intensity boost (jumps/year).  Sets clustering strength.
    beta_per_sec : float
        Exponential decay rate of excitation (per second).
        Half-life = ln(2) / beta_per_sec seconds.
    mu_j : float
        Mean log-jump size.
    sigma_j : float
        Std of log-jump size.
    """

    def __init__(
        self,
        heston: HestonSimulator | None = None,
        mu_per_year: float = 5_000.0,
        alpha_per_year: float = 88_000.0,
        beta_per_sec: float = 0.03,
        mu_j: float = 0.0,
        sigma_j: float = 0.001,
    ) -> None:
        self.heston = heston if heston is not None else HestonSimulator()
        self.mu_per_year = mu_per_year
        self.alpha_per_year = alpha_per_year
        self.beta_per_sec = beta_per_sec
        self.mu_j = mu_j
        self.sigma_j = sigma_j

        # Continuous-time branching ratio; warn if >= 1 (unstable)
        m = alpha_per_year / (beta_per_sec * SECS_PER_YEAR)
        if m >= 1.0:
            raise ValueError(
                f"Hawkes branching ratio m={m:.3f} >= 1 — process is explosive. "
                f"Reduce alpha_per_year or increase beta_per_sec."
            )

    def simulate(
        self,
        n_steps: int,
        dt_seconds: float,
        rng: np.random.Generator,
    ) -> SimulationResult:
        dt_years = dt_seconds / SECS_PER_YEAR
        decay = float(np.exp(-self.beta_per_sec * dt_seconds))

        base = self.heston.simulate(n_steps, dt_seconds, rng)
        log_s = base.log_price.copy()

        intensity = self.mu_per_year  # start at stationary baseline
        jump_indices: list[int] = []
        step_jump_sizes = np.zeros(n_steps + 1)

        for i in range(n_steps):
            # Jump probability from current intensity (exact Poisson approximation)
            prob = -np.expm1(-intensity * dt_years)  # = 1 − exp(−λ·dt)
            if rng.random() < prob:
                j_size = rng.normal(self.mu_j, max(self.sigma_j, 1e-12))
                step_jump_sizes[i + 1] = j_size
                jump_indices.append(i + 1)
                # Self-excite then decay
                intensity = (
                    self.mu_per_year
                    + (intensity - self.mu_per_year) * decay
                    + self.alpha_per_year
                )
            else:
                intensity = self.mu_per_year + (intensity - self.mu_per_year) * decay

        log_s += np.cumsum(step_jump_sizes)

        return SimulationResult(
            times=base.times,
            log_price=log_s,
            jump_indices=np.array(jump_indices, dtype=np.int64),
            sigma_path=base.sigma_path,
        )
