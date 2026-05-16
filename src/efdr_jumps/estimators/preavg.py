from __future__ import annotations

import numpy as np

from ..simulate.base import SECS_PER_YEAR

# Triangular weight function g(x) = min(x, 1-x) — continuous constants.
# Used for the noise-correction coefficient only (Jacod et al. 2009):
#   ψ₁ = ∫₀¹ (g'(x))² dx = 1   (g' = ±1 a.e.)  → appears in noise bias
#   ψ₂ = ∫₀¹ g(x)² dx    = 1/12 → appears in the denominator
_PSI1: float = 1.0
_PSI2: float = 1.0 / 12.0


def _triangular_weights(K: int) -> np.ndarray:
    """
    Triangular weights g_j = min((j+1)/K, (K-j)/K) for j=0,...,K-1.
    These are the discretised tent function values at mid-cell points.
    """
    j = np.arange(1, K + 1, dtype=np.float64)
    return np.minimum(j / K, (K + 1.0 - j) / K)


def preavg_rv(
    log_price: np.ndarray,
    dt_seconds: float,
    *,
    K: int = 20,
    noise_corrected: bool = False,
) -> float:
    """
    Pre-averaged realized variance (non-overlapping blocks, triangular weights).

    For each block k of size K, the pre-averaged return is
        r̄_k = Σⱼ g_j · r_{kK+j}

    Normalisation: annualised IV = Σ_k r̄_k² / (n_blocks · Δt · ||g||²)
    where ||g||² = Σ_j g_j² and Δt = dt_seconds / SECS_PER_YEAR.

    noise_corrected: if True, subtract an estimate of the residual noise bias
    using the continuous constant ψ₁ (requires σ_ε² ≈ Σ rᵢ² / (2n); rough
    and suitable only when the path is known to be very noisy).

    Returns
    -------
    Annualised integrated variance estimate (units: per year).
    """
    r = np.diff(log_price)
    n = len(r)
    n_blocks = n // K
    if n_blocks < 2:
        raise ValueError(f"Need ≥ 2 blocks; got {n_blocks} with K={K}, n={n}")

    g = _triangular_weights(K)  # (K,)
    g_sq_sum = float(np.dot(g, g))  # ||g||²

    r_trunc = r[: n_blocks * K].reshape(n_blocks, K)  # (n_blocks, K)
    r_bar = r_trunc @ g  # (n_blocks,)

    dt = dt_seconds / SECS_PER_YEAR
    raw = float(np.sum(r_bar**2)) / (n_blocks * dt * g_sq_sum)

    if noise_corrected:
        # Rough noise variance: σ̂²_ε ≈ Σ rᵢ² / (2n) (high-frequency noise estimator)
        sigma_eps_sq = float(np.sum(r[: n_blocks * K] ** 2)) / (2 * n_blocks * K)
        # Bias from noise in pre-averaged return: (2 σ²_ε / dt) × (||g||² - <g,shift(g)>)
        # Approximate using continuous constant ψ₁:
        noise_bias = _PSI1 * sigma_eps_sq / (dt * K)
        raw = max(raw - noise_bias, 0.0)

    return raw


def preavg_bv(
    log_price: np.ndarray,
    dt_seconds: float,
    *,
    K: int = 20,
) -> float:
    """
    Pre-averaged bipower variation (Podolskij & Vetter 2009).
    Computes BV on the sequence of pre-averaged returns.
    """
    r = np.diff(log_price)
    n = len(r)
    n_blocks = n // K
    if n_blocks < 3:
        raise ValueError(f"Need ≥ 3 blocks; got {n_blocks} with K={K}, n={n}")

    g = _triangular_weights(K)
    g_sq_sum = float(np.dot(g, g))

    r_trunc = r[: n_blocks * K].reshape(n_blocks, K)
    r_bar = r_trunc @ g  # (n_blocks,)

    # BV on pre-averaged returns (π/2 scale factor, finite-sample correction)
    m = len(r_bar)
    bv_raw = (np.pi / 2.0) * (m / (m - 1)) * float(np.sum(np.abs(r_bar[:-1]) * np.abs(r_bar[1:])))
    dt = dt_seconds / SECS_PER_YEAR
    return bv_raw / (n_blocks * dt * g_sq_sum)
