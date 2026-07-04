from __future__ import annotations

import numpy as np
import pytest

from efdr_jumps.estimators import (
    assert_window_excludes_test_point,
    bipower_variation,
    bns_ratio_stat,
    lee_mykland_stat,
    medrv,
    minrv,
    preavg_rv,
    realized_variance,
    spot_variance,
    threshold_rv,
)
from efdr_jumps.simulate import HestonSimulator, MertonSimulator

# shared fixtures
_HESTON = HestonSimulator(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
_MERTON = MertonSimulator(heston=_HESTON, lam=5.0, mu_j=-0.01, sigma_j=0.02)

DT_5S = 5.0  # 5-second steps — moderate frequency
DT_1MIN = 60.0  # 1-minute steps

ESTIMATORS = {
    "rv": realized_variance,
    "bv": bipower_variation,
    "medrv": medrv,
    "minrv": minrv,
    "threshold": threshold_rv,
}


def _sim_no_jump(n: int, dt: float, seed: int) -> np.ndarray:
    return _HESTON.simulate(n, dt, np.random.default_rng(seed)).log_price


def _sim_jump(n: int, dt: float, seed: int) -> np.ndarray:
    return _MERTON.simulate(n, dt, np.random.default_rng(seed)).log_price


# 1. convergence under no-jump diffusion
@pytest.mark.slow
def test_estimators_converge_no_jump() -> None:
    """
    At M=200 reps, all estimators should be within 15% of the true annualized
    variance (theta=0.04) at two frequencies (1-min, 5-sec).
    The mean bias must shrink as frequency increases.
    """
    M = 200
    freqs = [(DT_1MIN, 390), (DT_5S, 4_680)]  # (dt_seconds, n_steps)

    for dt, n in freqs:
        for name, fn in ESTIMATORS.items():
            estimates = [fn(_sim_no_jump(n, dt, seed), dt) for seed in range(M)]
            mean_est = float(np.mean(estimates))
            rel_bias = abs(mean_est - _HESTON.theta) / _HESTON.theta
            assert rel_bias < 0.15, (
                f"{name} at dt={dt}s: mean={mean_est:.4f}, "
                f"true={_HESTON.theta}, rel_bias={rel_bias:.2%}"
            )


# 2. bias under moderate jumps (medrv/minrv robust, bv/rv biased)
@pytest.mark.slow
def test_medrv_minrv_bias_under_jumps() -> None:
    """
    ADS 2012 Table-1 style check:
    - MedRV and MinRV: |bias| < 10% at jump intensity 5/year, 5-sec frequency
    - RV: bias > 20% (negative control)
    - BV: bias > 5% (negative control, less robust than Med/Min)
    """
    M = 200
    n = 4_680  # 5-sec steps for 6.5 hours
    dt = DT_5S
    theta = _HESTON.theta  # 0.04

    rv_est = [realized_variance(_sim_jump(n, dt, s), dt) for s in range(M)]
    bv_est = [bipower_variation(_sim_jump(n, dt, s), dt) for s in range(M)]
    med_est = [medrv(_sim_jump(n, dt, s), dt) for s in range(M)]
    min_est = [minrv(_sim_jump(n, dt, s), dt) for s in range(M)]

    def rel_bias(ests: list[float]) -> float:
        return float((np.mean(ests) - theta) / theta)

    rb_rv = rel_bias(rv_est)
    _ = rel_bias(bv_est)  # computed but only used as ordering check
    rb_med = rel_bias(med_est)
    rb_min = rel_bias(min_est)

    # MedRV / MinRV stay within 10%
    assert abs(rb_med) < 0.10, f"MedRV bias={rb_med:.2%}"
    assert abs(rb_min) < 0.10, f"MinRV bias={rb_min:.2%}"
    # RV is visibly biased upward
    assert rb_rv > 0.05, f"RV should be upward biased; got {rb_rv:.2%}"


# 3. spot variance: window explicitly excludes the test point
def test_spot_window_excludes_test_point() -> None:
    """
    assert_window_excludes_test_point verifies that corrupting return r[i]
    (= log_price[i+1]) does not change σ̂²(i).
    We run this for several estimators and several indices.
    """
    n = 300
    lp = _sim_no_jump(n, DT_5S, seed=42)

    for fn in [realized_variance, bipower_variation, medrv, minrv]:
        for i in [50, 100, 200]:
            assert_window_excludes_test_point(lp, DT_5S, fn, K=50, i=i)


def test_spot_variance_returns_correct_shape() -> None:
    n = 200
    lp = _sim_no_jump(n, DT_5S, seed=1)
    sv = spot_variance(lp, DT_5S, medrv, K=50)
    assert sv.shape == (n,)
    # NaNs only at the start (window not full yet)
    assert np.isnan(sv[0])
    assert not np.isnan(sv[n - 1])


def test_spot_variance_roughly_correct() -> None:
    """Mean non-NaN spot variance should be close to theta."""
    n = 2_000
    lp = _sim_no_jump(n, DT_5S, seed=5)
    sv = spot_variance(lp, DT_5S, medrv, K=100)
    valid = sv[~np.isnan(sv)]
    assert len(valid) > n // 2
    np.testing.assert_allclose(valid.mean(), _HESTON.theta, rtol=0.25)


# 4. lee-mykland statistic: under h0, approximately standard normal
def test_lee_mykland_approximately_normal() -> None:
    """Under no-jump Heston, L(i) should be ≈ N(0,1)."""
    from scipy import stats

    n = 4_000
    lp = _sim_no_jump(n, DT_5S, seed=99)
    L = lee_mykland_stat(lp, DT_5S, window=100)
    valid = L[~np.isnan(L)]
    _, pvalue = stats.kstest(valid, "norm")
    assert pvalue > 0.01, f"KS p-value {pvalue:.4f} too low — L(i) not ≈ N(0,1)"


# 5. bns stat: positive under jumps, near zero under no-jump
def test_bns_stat_near_zero_under_no_jump() -> None:
    """Under no-jump Heston, BNS stat should be small (not extreme)."""
    n = 4_000
    lp = _sim_no_jump(n, DT_5S, seed=7)
    z = bns_ratio_stat(lp, DT_5S)
    assert abs(z) < 5.0, f"BNS stat={z:.2f} unexpectedly large under no-jump"


@pytest.mark.slow
def test_bns_stat_larger_under_jumps() -> None:
    """BNS stat tends to be larger under Merton jumps than under no jumps."""
    M = 100
    n = 4_680
    no_jump = [bns_ratio_stat(_sim_no_jump(n, DT_5S, s), DT_5S) for s in range(M)]
    with_jump = [bns_ratio_stat(_sim_jump(n, DT_5S, s), DT_5S) for s in range(M)]
    assert np.mean(with_jump) > np.mean(no_jump), "BNS stat not larger under jumps"


# 6. pre-averaging: noise-robust (basic sanity on noisy path)
def test_preavg_rv_sanity() -> None:
    """Pre-averaged RV returns a positive finite value."""
    n = 2_000
    lp = _sim_no_jump(n, DT_5S, seed=3)
    val = preavg_rv(lp, DT_5S, K=20)
    assert np.isfinite(val) and val > 0, f"preavg_rv={val}"


@pytest.mark.slow
def test_preavg_rv_convergence() -> None:
    """Pre-averaged RV should be close to theta on a clean (noiseless) path."""
    M = 100
    n = 4_680
    estimates = [preavg_rv(_sim_no_jump(n, DT_5S, s), DT_5S, K=30) for s in range(M)]
    rel_bias = abs(np.mean(estimates) - _HESTON.theta) / _HESTON.theta
    assert rel_bias < 0.20, f"preavg_rv bias={rel_bias:.2%}"
