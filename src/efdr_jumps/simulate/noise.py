from __future__ import annotations

import numpy as np

from .base import PathSimulator, SimulationResult


class NoisyPath(PathSimulator):
    """
    Wraps any PathSimulator and adds microstructure noise to the observed path.

    Two models are supported:
    - "additive" : ε_i ~ N(0, σ_noise²)  — standard additive MMN model
    - "one_sided": ε_i ~ Exp(1/σ_noise)  — one-sided à la Bibinger (2024),
                   modeling bid-ask spread noise where prices are always quoted
                   above the efficient price (ask side).

    Jump indices and sigma_path are passed through unchanged — they refer to the
    underlying true process, which is what the detector is trying to recover.

    Parameters
    ----------
    simulator   : underlying PathSimulator
    noise_type  : "additive" or "one_sided"
    sigma_noise : noise scale (std for additive, mean for one_sided), in log-price units
    """

    def __init__(
        self,
        simulator: PathSimulator,
        noise_type: str = "additive",
        sigma_noise: float = 1e-4,
    ) -> None:
        if noise_type not in {"additive", "one_sided"}:
            raise ValueError(f"noise_type must be 'additive' or 'one_sided', got {noise_type!r}")
        self.simulator = simulator
        self.noise_type = noise_type
        self.sigma_noise = sigma_noise

    def simulate(
        self,
        n_steps: int,
        dt_seconds: float,
        rng: np.random.Generator,
    ) -> SimulationResult:
        result = self.simulator.simulate(n_steps, dt_seconds, rng)
        noise = self._draw_noise(n_steps + 1, rng)
        return SimulationResult(
            times=result.times,
            log_price=result.log_price + noise,
            jump_indices=result.jump_indices,
            sigma_path=result.sigma_path,
        )

    def _draw_noise(self, size: int, rng: np.random.Generator) -> np.ndarray:
        if self.noise_type == "additive":
            return rng.normal(0.0, self.sigma_noise, size)
        # one_sided: Exp with mean sigma_noise; noise is always non-negative
        return rng.exponential(self.sigma_noise, size)
