from __future__ import annotations

import numpy as np


def construct_evalue(
    r_i: float,
    sigma_hat_i: float,
    dt_years: float,
    tau_factor: float = 5.0,
) -> float:
    """
    GROW e-value for jump detection using Gaussian mixture prior.

    Closed-form solution to the mixture e-value integral:
        E_i = ∫ exp(J·r_i / Δσ̂² - J² / (2·Δσ̂²)) dπ(J)
    where π(J) = N(0, τ²) is a Gaussian prior.

    Using the formula ∫ exp(ax - bx²) dx = √(π/b) exp(a²/(4b)), this yields:
        E_i = √(Δσ̂² / (Δσ̂² + τ²)) · exp(r_i² · τ² / (2·Δσ̂²·(Δσ̂² + τ²)))

    Parameters
    ----------
    r_i : float
        Return (log-price increment) at time i
    sigma_hat_i : float
        Spot volatility estimate σ̂_i (annualized, sqrt(variance))
    dt_years : float
        Time increment dt in years (e.g. 5 seconds / (252*6.5*3600) ≈ 1.98e-6)
    tau_factor : float, default=5.0
        Prior scale multiplier: τ = tau_factor × σ̂ × √dt.
        Scaling by √dt ensures dimensional consistency in discrete time.
        tau_factor=5.0 balances e-power with validity under H0.

    Returns
    -------
    float
        E-value E_i ≥ 0. Under H0 (no jump), E[E_i] ≤ 1 for validity.
    """
    dv = dt_years * sigma_hat_i**2  # Δσ̂²
    tau = tau_factor * sigma_hat_i * np.sqrt(dt_years)  # Prior scale (scaled by √dt for unit consistency)
    tau2 = tau**2  # τ²

    numerator = dv / (dv + tau2)
    exponent = r_i**2 * tau2 / (2 * dv * (dv + tau2))

    e_value = np.sqrt(numerator) * np.exp(exponent)
    return float(e_value)
