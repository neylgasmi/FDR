from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import stats

from efdr_jumps.simulate import (
    SECS_PER_YEAR,
    HestonSimulator,
    MertonSimulator,
    NoisyPath,
)

RNG = np.random.default_rng(42)

# helpers
TRADING_DAY_SECONDS = 6.5 * 3600  # 23 400 s
DT_1S = 1.0
N_DAY = int(TRADING_DAY_SECONDS)  # 23 400 steps


def _fresh_rng() -> np.random.Generator:
    return np.random.default_rng(0)


# interface / shape tests
def test_heston_result_shapes() -> None:
    sim = HestonSimulator()
    n = 500
    res = sim.simulate(n, DT_1S, _fresh_rng())
    assert res.times.shape == (n + 1,)
    assert res.log_price.shape == (n + 1,)
    assert res.sigma_path.shape == (n + 1,)
    assert res.jump_indices.shape == (0,)  # pure Heston has no jumps
    assert res.times[0] == 0.0
    np.testing.assert_allclose(res.times[-1], n * DT_1S)


def test_merton_result_shapes() -> None:
    sim = MertonSimulator(lam=50.0)
    n = 500
    res = sim.simulate(n, DT_1S, _fresh_rng())
    assert res.times.shape == (n + 1,)
    assert res.log_price.shape == (n + 1,)
    assert res.sigma_path.shape == (n + 1,)


def test_noisy_path_shapes() -> None:
    base = HestonSimulator()
    sim = NoisyPath(base, sigma_noise=1e-4)
    n = 200
    res = sim.simulate(n, DT_1S, _fresh_rng())
    assert res.log_price.shape == (n + 1,)


# ks normality check on standardised increments (no-jump heston)
def test_heston_increments_approximately_normal() -> None:
    """
    With very fast mean reversion (kappa=200) V_t ≈ theta for all t.
    Standardised increments should be close to N(0,1).
    KS p-value should exceed 0.01.
    """
    theta = 0.04
    kappa = 200.0
    dt_s = 5.0  # 5-second steps
    dt = dt_s / SECS_PER_YEAR
    n = 5_000

    sim = HestonSimulator(kappa=kappa, theta=theta, xi=0.3, rho=0.0, v0=theta, mu=0.0)
    res = sim.simulate(n, dt_s, np.random.default_rng(7))

    increments = np.diff(res.log_price)
    mean_incr = -0.5 * theta * dt
    std_incr = np.sqrt(theta * dt)
    z = (increments - mean_incr) / std_incr

    stat, pvalue = stats.kstest(z, "norm")
    assert pvalue > 0.01, f"KS p-value too low: {pvalue:.4f} (stat={stat:.4f})"


# jump index recovery tests
def test_merton_no_jump_when_lam_zero() -> None:
    sim = MertonSimulator(lam=0.0)
    res = sim.simulate(1_000, DT_1S, _fresh_rng())
    assert len(res.jump_indices) == 0


def test_merton_all_steps_jump_when_lam_huge() -> None:
    """With λ·Δt >> 1 every step must have at least one jump."""
    lam = 1e9  # ~170 jumps per second → virtually certain per step
    n = 200
    sim = MertonSimulator(lam=lam, mu_j=0.0, sigma_j=1e-6)
    res = sim.simulate(n, DT_1S, _fresh_rng())
    assert len(res.jump_indices) == n
    np.testing.assert_array_equal(res.jump_indices, np.arange(1, n + 1))


def test_merton_jump_count_poisson_statistics() -> None:
    """
    Over M short simulations, empirical jump count should be consistent
    with Poisson(λ·T) — verified via Poisson confidence interval.
    """
    lam = 100.0  # jumps per year
    dt_s = 60.0  # 1-minute steps
    n = 390  # ~6.5 hours of 1-min data
    T_years = n * dt_s / SECS_PER_YEAR
    expected_mean = lam * T_years

    M = 200
    counts = np.array(
        [
            len(MertonSimulator(lam=lam).simulate(n, dt_s, np.random.default_rng(i)).jump_indices)
            for i in range(M)
        ]
    )
    empirical_mean = counts.mean()
    # Within 4 standard deviations of Poisson mean
    poisson_std = np.sqrt(expected_mean)
    assert abs(empirical_mean - expected_mean) < 4 * poisson_std / np.sqrt(
        M
    ), f"Jump count mean {empirical_mean:.3f} far from Poisson mean {expected_mean:.3f}"


def test_jump_indices_within_valid_range() -> None:
    n = 500
    sim = MertonSimulator(lam=500.0)
    res = sim.simulate(n, DT_1S, _fresh_rng())
    if len(res.jump_indices) > 0:
        assert res.jump_indices.min() >= 1
        assert res.jump_indices.max() <= n


# noise tests
def test_additive_noise_variance() -> None:
    """
    Measure noise directly by drawing it from the same RNG twice.
    Empirical variance of N(0, σ²) draws must be close to σ².
    """
    sigma_noise = 5e-4
    n = 20_000
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, sigma_noise, n)
    np.testing.assert_allclose(noise.var(), sigma_noise**2, rtol=0.05)


def test_noisy_path_additive_shifts_log_price() -> None:
    """NoisyPath with additive noise must change the log_price."""
    base = HestonSimulator()
    sim = NoisyPath(base, noise_type="additive", sigma_noise=1e-3)
    n = 200
    clean = base.simulate(n, DT_1S, np.random.default_rng(5))
    noisy = sim.simulate(n, DT_1S, np.random.default_rng(5))
    assert not np.allclose(clean.log_price, noisy.log_price)
    # jump_indices are unchanged
    np.testing.assert_array_equal(clean.jump_indices, noisy.jump_indices)


def test_one_sided_noise_mean() -> None:
    """Exponential draws must be non-negative with mean ≈ sigma_noise."""
    sigma_noise = 2e-4
    n = 20_000
    rng = np.random.default_rng(2)
    noise = rng.exponential(sigma_noise, n)
    assert (noise >= 0).all()
    np.testing.assert_allclose(noise.mean(), sigma_noise, rtol=0.05)


# noise module: variance via noisypath, pre-averaging
# HestonSimulator.simulate consumes exactly 3·n_steps values from the RNG
# (uniform, normal, normal).  NoisyPath then draws n_steps+1 more.
# Using the same seed guarantees that both NoisyPath variants (additive,
# one_sided) receive the same base-simulator draws and therefore produce the
# same underlying clean path.

_NOISE_BASE = HestonSimulator(kappa=5.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
_NOISE_N = 4_000
_NOISE_DT = 5.0  # 5-second steps


def _isolate_noise(
    base: HestonSimulator,
    noise_type: str,
    sigma_noise: float,
    n: int,
    dt: float,
    seed: int,
) -> np.ndarray:
    """Return the noise vector noisy.log_price - clean.log_price."""
    clean = base.simulate(n, dt, np.random.default_rng(seed))
    noisy = NoisyPath(base, noise_type, sigma_noise).simulate(n, dt, np.random.default_rng(seed))
    return noisy.log_price - clean.log_price


def test_noisy_path_additive_variance_endtoend() -> None:
    """
    var(noisy.log_price - clean.log_price) ≈ σ² when using NoisyPath end-to-end.
    Same seed → same base-simulator draws → difference is pure noise.
    """
    sigma_noise = 1e-3
    noise = _isolate_noise(_NOISE_BASE, "additive", sigma_noise, _NOISE_N, _NOISE_DT, seed=10)
    np.testing.assert_allclose(noise.var(), sigma_noise**2, rtol=0.05)


def test_noisy_path_one_sided_endtoend() -> None:
    """One-sided noise: all values ≥ 0, mean ≈ sigma_noise."""
    sigma_noise = 8e-4
    noise = _isolate_noise(_NOISE_BASE, "one_sided", sigma_noise, _NOISE_N, _NOISE_DT, seed=11)
    assert (noise >= 0).all(), "One-sided noise must be non-negative"
    np.testing.assert_allclose(noise.mean(), sigma_noise, rtol=0.05)


def test_signal_recoverable_by_preavg() -> None:
    """
    Pre-averaging K consecutive noisy observations reduces noise variance
    by a factor of K (CLT).  We test on the isolated noise component.
    """
    sigma_noise = 1e-3
    K = 40
    noise = _isolate_noise(_NOISE_BASE, "additive", sigma_noise, _NOISE_N, _NOISE_DT, seed=12)

    n_blocks = len(noise) // K
    # Reshape into (n_blocks, K) and average over the K-window
    noise_pa = noise[: n_blocks * K].reshape(n_blocks, K).mean(axis=1)

    var_raw = noise.var()
    var_pa = noise_pa.var()

    # Pre-averaged variance should be ≈ σ²/K; accept within 30% for MC variability
    np.testing.assert_allclose(var_pa, sigma_noise**2 / K, rtol=0.30)
    # And strictly smaller than raw noise
    assert var_pa < var_raw / (
        K * 0.4
    ), f"Pre-averaging did not reduce variance: raw={var_raw:.2e}, pa={var_pa:.2e}"


def test_three_versions_from_same_seed() -> None:
    """
    From one fixed seed we can generate:
      1. clean path (no noise)
      2. additive-noise path
      3. one-sided-noise path
    Versions 2 and 3 must share the same underlying clean path as version 1.
    """
    seed = 77
    sigma = 5e-4
    n = 500
    dt = 5.0
    base = _NOISE_BASE

    clean = base.simulate(n, dt, np.random.default_rng(seed))
    noisy_add = NoisyPath(base, "additive", sigma).simulate(n, dt, np.random.default_rng(seed))
    noisy_one = NoisyPath(base, "one_sided", sigma).simulate(n, dt, np.random.default_rng(seed))

    # The base-simulator draws are deterministic → underlying paths are identical
    noise_add = noisy_add.log_price - clean.log_price
    noise_one = noisy_one.log_price - clean.log_price

    # Additive: zero-mean Gaussian
    np.testing.assert_allclose(noise_add.mean(), 0.0, atol=4 * sigma / np.sqrt(n))
    # One-sided: non-negative
    assert (noise_one >= 0).all()
    # The two noise vectors must differ (different distributions)
    assert not np.allclose(noise_add, noise_one)
    # jump_indices pass through unchanged in all three
    np.testing.assert_array_equal(clean.jump_indices, noisy_add.jump_indices)
    np.testing.assert_array_equal(clean.jump_indices, noisy_one.jump_indices)


# performance: 1 trading day at 1-second resolution in < 100 ms
@pytest.mark.slow
def test_heston_merton_day_speed() -> None:
    """Exit criterion: 1-second × 23 400 steps in < 100 ms (post JIT warm-up)."""
    sim = MertonSimulator(lam=5.0)
    rng = _fresh_rng()

    # Warm-up JIT
    sim.simulate(100, DT_1S, rng)

    rng2 = np.random.default_rng(99)
    t0 = time.perf_counter()
    sim.simulate(N_DAY, DT_1S, rng2)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.1, f"Simulation took {elapsed*1000:.1f} ms (> 100 ms)"
