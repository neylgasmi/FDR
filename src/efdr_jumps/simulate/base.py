from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# 1 year = 252 trading days × 6.5 hours × 3600 seconds
SECS_PER_YEAR: float = 252.0 * 6.5 * 3600.0


@dataclass
class SimulationResult:
    """Output of any PathSimulator.simulate() call."""

    times: np.ndarray       # (N+1,)  seconds since market open
    log_price: np.ndarray   # (N+1,)  natural log of price
    jump_indices: np.ndarray  # (K,)  indices into `times` where jumps land
    sigma_path: np.ndarray  # (N+1,)  annualized spot volatility


class PathSimulator(ABC):
    @abstractmethod
    def simulate(
        self,
        n_steps: int,
        dt_seconds: float,
        rng: np.random.Generator,
    ) -> SimulationResult:
        """
        Parameters
        ----------
        n_steps     : number of time steps
        dt_seconds  : step size in seconds
        rng         : seeded Generator for reproducibility
        """
