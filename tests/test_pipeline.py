from __future__ import annotations

import numpy as np
import pytest

from efdr_jumps.pipeline.detector import (
    EstimatorConfig,
    FDRConfig,
    SimulatorConfig,
    run_detector,
)
from efdr_jumps.pipeline.metrics import (
    aggregate,
    detection_delays,
    f1_score,
    fdp,
    power,
    rejection_rate,
)

# --------------------------------------------------------------------------- #
# Metrics unit tests                                                           #
# --------------------------------------------------------------------------- #


def test_fdp_empty_rejection():
    assert fdp(frozenset(), frozenset({1, 2})) == 0.0


def test_fdp_all_false():
    assert fdp(frozenset({3, 4}), frozenset({1, 2})) == 1.0


def test_fdp_all_true():
    assert fdp(frozenset({1, 2}), frozenset({1, 2})) == 0.0


def test_fdp_mixed():
    # 1 FP, 1 TP → FDP = 0.5
    assert fdp(frozenset({1, 99}), frozenset({1})) == pytest.approx(0.5)


def test_power_exact():
    # True jump at 5, rejection at 5 → power = 1
    assert power(frozenset({5}), frozenset({5}), k=0) == pytest.approx(1.0)


def test_power_within_tolerance():
    # True jump at 5, rejection at 7, k=2 → detected
    assert power(frozenset({7}), frozenset({5}), k=2) == pytest.approx(1.0)


def test_power_outside_tolerance():
    assert power(frozenset({10}), frozenset({5}), k=2) == pytest.approx(0.0)


def test_power_no_true_jumps():
    assert np.isnan(power(frozenset({1}), frozenset(), k=2))


def test_f1_perfect():
    rj = frozenset({5, 10})
    tj = frozenset({5, 10})
    assert f1_score(rj, tj, k=0) == pytest.approx(1.0)


def test_f1_no_overlap():
    assert f1_score(frozenset({1}), frozenset({100}), k=0) == pytest.approx(0.0)


def test_f1_both_empty():
    assert f1_score(frozenset(), frozenset(), k=2) == pytest.approx(1.0)


def test_detection_delays_basic():
    # Jump at 5, rejection at 6 → delay = 1
    d = detection_delays(frozenset({6}), frozenset({5}), k=2)
    assert d["mean"] == pytest.approx(1.0)
    assert d["median"] == pytest.approx(1.0)


def test_detection_delays_no_hit():
    d = detection_delays(frozenset({100}), frozenset({5}), k=2)
    assert np.isnan(d["mean"])


def test_rejection_rate():
    assert rejection_rate(frozenset({1, 2, 3}), 100) == pytest.approx(0.03)
    assert rejection_rate(frozenset(), 100) == pytest.approx(0.0)


def test_aggregate_basic():
    runs = [
        {
            "fdp": 0.0,
            "power": 1.0,
            "f1": 1.0,
            "delay_mean": 0.0,
            "delay_median": 0.0,
            "rej_rate": 0.1,
        },
        {
            "fdp": 0.1,
            "power": 0.8,
            "f1": 0.9,
            "delay_mean": 1.0,
            "delay_median": 1.0,
            "rej_rate": 0.05,
        },
    ]
    agg = aggregate(runs)
    assert agg["fdp"] == pytest.approx(0.05)
    assert agg["power"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Detector smoke tests                                                         #
# --------------------------------------------------------------------------- #


def test_run_detector_heston_h0():
    """Pure diffusion run: rejection set must be a frozenset, FDP=0 trivially."""
    sim_cfg = SimulatorConfig(regime="heston_h0", n_steps=100, dt_seconds=5.0)
    est_cfg = EstimatorConfig(kind="medrv", window=30)
    fdr_cfg = FDRConfig(algo="elond", alpha=0.05)

    result = run_detector(sim_cfg, est_cfg, fdr_cfg, seed=0)
    assert isinstance(result.rejection_set, frozenset)
    assert result.n_hypotheses == 100
    assert result.wall_clock > 0.0
    assert len(result.true_jumps) == 0  # H0: no jumps
    assert fdp(result.rejection_set, result.true_jumps) == 0.0


def test_run_detector_moderate_ebh():
    """Moderate regime, e-BH: check types and metric bounds."""
    sim_cfg = SimulatorConfig(regime="moderate", n_steps=200)
    est_cfg = EstimatorConfig(kind="medrv", window=50)
    fdr_cfg = FDRConfig(algo="ebh", alpha=0.1)

    result = run_detector(sim_cfg, est_cfg, fdr_cfg, seed=1)
    assert isinstance(result.rejection_set, frozenset)
    assert result.n_hypotheses == 200
    assert 0.0 <= fdp(result.rejection_set, result.true_jumps) <= 1.0
    assert 0.0 <= rejection_rate(result.rejection_set, result.n_hypotheses) <= 1.0


def test_run_detector_all_algos_smoke():
    """All 7 algorithms run without error on a short path."""
    algos = ["ebh", "bh_lm", "bh_bns", "elond", "elord", "esaffron", "stopped_ebh"]
    sim_cfg = SimulatorConfig(regime="moderate", n_steps=80)
    est_cfg = EstimatorConfig(kind="medrv", window=20)

    for algo in algos:
        fdr_cfg = FDRConfig(algo=algo, alpha=0.1)
        result = run_detector(sim_cfg, est_cfg, fdr_cfg, seed=42)
        assert isinstance(result.rejection_set, frozenset), f"{algo}: bad rejection_set type"
        assert result.n_hypotheses == 80, f"{algo}: wrong n_hypotheses"


def test_run_detector_hawkes_dense():
    """hawkes_dense regime runs all 7 algos and produces more jumps than moderate."""
    algos = ["ebh", "bh_lm", "bh_bns", "elond", "elord", "esaffron", "stopped_ebh"]
    sim_hawkes = SimulatorConfig(regime="hawkes_dense", n_steps=200, dt_seconds=5.0)
    est_cfg = EstimatorConfig(kind="medrv", window=50)

    for algo in algos:
        fdr_cfg = FDRConfig(algo=algo, alpha=0.1)
        result = run_detector(sim_hawkes, est_cfg, fdr_cfg, seed=0)
        assert isinstance(result.rejection_set, frozenset), f"{algo}: bad type"
        assert result.n_hypotheses == 200

    # Hawkes should produce strictly more jumps on average than moderate
    n_hawkes = [
        len(run_detector(sim_hawkes, est_cfg, FDRConfig(algo="elond"), seed=s).true_jumps)
        for s in range(30)
    ]
    sim_mod = SimulatorConfig(regime="moderate", n_steps=200, dt_seconds=5.0)
    n_mod = [
        len(run_detector(sim_mod, est_cfg, FDRConfig(algo="elond"), seed=s).true_jumps)
        for s in range(30)
    ]
    import numpy as np

    assert np.mean(n_hawkes) > np.mean(n_mod), (
        f"hawkes_dense should have more jumps than moderate: "
        f"hawkes={np.mean(n_hawkes):.2f} mod={np.mean(n_mod):.2f}"
    )


def test_run_detector_result_traceable():
    """DetectorResult embeds full config for traceability."""
    sim_cfg = SimulatorConfig(regime="rare", n_steps=100, jump_size_sigma=5.0)
    est_cfg = EstimatorConfig(kind="medrv")
    fdr_cfg = FDRConfig(algo="elond", alpha=0.05)

    result = run_detector(sim_cfg, est_cfg, fdr_cfg, seed=7)
    assert result.sim_config.regime == "rare"
    assert result.fdr_config.algo == "elond"
    assert result.seed == 7


@pytest.mark.slow
def test_pipeline_fdr_validity_moderate():
    """FDR ≤ alpha + 2·SE over M=50 moderate-regime runs (e-LOND)."""
    M = 50
    alpha = 0.05
    sim_cfg = SimulatorConfig(regime="moderate", n_steps=300)
    est_cfg = EstimatorConfig(kind="medrv", window=50)
    fdr_cfg = FDRConfig(algo="elond", alpha=alpha)

    fdps = []
    for seed in range(M):
        result = run_detector(sim_cfg, est_cfg, fdr_cfg, seed=seed)
        fdps.append(fdp(result.rejection_set, result.true_jumps))

    fdr_empirical = float(np.mean(fdps))
    se = float(np.std(fdps) / np.sqrt(M))
    assert (
        fdr_empirical <= alpha + 2 * se
    ), f"e-LOND pipeline FDR={fdr_empirical:.4f} > alpha+2SE={alpha + 2*se:.4f}"


@pytest.mark.slow
def test_pipeline_at_least_one_rejection():
    """At least one algorithm makes at least one rejection on the moderate regime."""
    algos = ["ebh", "elond", "elord", "esaffron", "stopped_ebh"]
    sim_cfg = SimulatorConfig(regime="moderate", n_steps=500, jump_size_sigma=5.0)
    est_cfg = EstimatorConfig(kind="medrv", window=50)

    any_rejection = False
    for seed in range(20):
        for algo in algos:
            fdr_cfg = FDRConfig(algo=algo, alpha=0.1)
            result = run_detector(sim_cfg, est_cfg, fdr_cfg, seed=seed)
            if result.rejection_set:
                any_rejection = True
                break
        if any_rejection:
            break

    assert any_rejection, "No algorithm made any rejection on 20 moderate paths — power failure"
