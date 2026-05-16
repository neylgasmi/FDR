from __future__ import annotations

import numba
import numpy as np

from .base import SECS_PER_YEAR, PathSimulator, SimulationResult


@numba.njit(cache=True)
def _qe_kernel(
    log_s: np.ndarray,
    v: np.ndarray,
    U: np.ndarray,
    Z_v: np.ndarray,
    Z_s: np.ndarray,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    mu: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Andersen (2008) Quadratic Exponential scheme for Heston SV.

    Mutates log_s and v in-place. dt is in years.
    Reference: Andersen (2008) §3, eqs. (17), (27)-(29).
    """
    PSI_C = 1.5

    # Variance conditional moment coefficients — hoisted outside the loop
    e = np.exp(-kappa * dt)
    m_const = theta * (1.0 - e)  # theta contribution to E[V_{t+dt}|V_t]
    s2_c1 = xi * xi * e / kappa * (1.0 - e)  # V_t coefficient in Var
    s2_c2 = theta * xi * xi / (2.0 * kappa) * (1.0 - e) ** 2  # constant in Var

    # Log-price coefficients with γ₁ = γ₂ = 0.5 (Andersen eq. 29)
    K0 = -rho * kappa * theta * dt / xi
    K1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
    K2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
    K34 = 0.5 * dt * (1.0 - rho * rho)  # K3 = K4 by symmetry

    n = len(U)
    for i in range(n):
        vi = v[i]
        m = m_const + vi * e
        s2 = s2_c1 * vi + s2_c2
        psi = s2 / (m * m + 1e-14)  # guard against m ≈ 0

        if psi < 1e-12:
            vn = m  # degenerate: variance is deterministic (ξ ≈ 0)
        elif psi <= PSI_C:
            inv_psi = 1.0 / psi
            # b² from matching first two moments (Andersen eq. 27)
            b2 = 2.0 * inv_psi - 1.0 + np.sqrt(2.0 * inv_psi * (2.0 * inv_psi - 1.0))
            a = m / (1.0 + b2)
            vn = a * (np.sqrt(b2) + Z_v[i]) ** 2
        else:
            p = (psi - 1.0) / (psi + 1.0)
            beta = 2.0 / (m * (psi + 1.0))
            u = min(U[i], 1.0 - 1e-14)  # guard against u = 1
            if u <= p:
                vn = 0.0
            else:
                vn = np.log((1.0 - p) / (1.0 - u)) / beta

        v[i + 1] = vn

        int_var = K34 * (vi + vn)
        log_s[i + 1] = (
            log_s[i] + mu * dt + K0 + K1 * vi + K2 * vn + np.sqrt(max(int_var, 0.0)) * Z_s[i]
        )

    return log_s, v


class HestonSimulator(PathSimulator):
    """
    Heston stochastic volatility model with Andersen (2008) QE scheme.

    Parameters
    ----------
    kappa : mean-reversion speed of variance
    theta : long-run variance (σ_∞² in annualized units)
    xi    : vol-of-vol
    rho   : price-variance correlation (leverage)
    v0    : initial variance
    mu    : log-price drift (annualized); set 0 for pure diffusion tests
    """

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.5,
        rho: float = -0.7,
        v0: float = 0.04,
        mu: float = 0.0,
    ) -> None:
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
        self.mu = mu

    def simulate(
        self,
        n_steps: int,
        dt_seconds: float,
        rng: np.random.Generator,
    ) -> SimulationResult:
        dt = dt_seconds / SECS_PER_YEAR

        log_s = np.zeros(n_steps + 1)
        v = np.zeros(n_steps + 1)
        v[0] = self.v0

        U = rng.uniform(0.0, 1.0, n_steps)
        Z_v = rng.standard_normal(n_steps)
        Z_s = rng.standard_normal(n_steps)

        log_s, v = _qe_kernel(
            log_s,
            v,
            U,
            Z_v,
            Z_s,
            self.kappa,
            self.theta,
            self.xi,
            self.rho,
            self.mu,
            dt,
        )

        times = np.arange(n_steps + 1, dtype=np.float64) * dt_seconds
        return SimulationResult(
            times=times,
            log_price=log_s,
            jump_indices=np.empty(0, dtype=np.int64),
            sigma_path=np.sqrt(v),  # annualized: v already in per-year units
        )
