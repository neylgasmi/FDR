"""
Monte Carlo power grid for efdr-jumps.

Usage:
    python experiments/02_power_grid.py --config experiments/configs/grid_quick.yaml
    python experiments/02_power_grid.py --config experiments/configs/grid_main.yaml \\
        --output results/grid_main.parquet --n-jobs -1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

# Ensure the src package is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from efdr_jumps.pipeline.detector import (
    DetectorResult,
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
# Task definition                                                              #
# --------------------------------------------------------------------------- #


def _make_task(
    algo: str,
    estimator: str,
    dt_seconds: float,
    regime: str,
    jump_size_sigma: float,
    alpha: float,
    w1: float,
    seed: int,
) -> tuple[SimulatorConfig, EstimatorConfig, FDRConfig, int]:
    sim_cfg = SimulatorConfig(
        regime=regime,
        dt_seconds=dt_seconds,
        n_steps=500,
        jump_size_sigma=jump_size_sigma,
    )
    est_cfg = EstimatorConfig(kind=estimator, window=50)
    fdr_cfg = FDRConfig(algo=algo, alpha=alpha, w1=w1)
    return sim_cfg, est_cfg, fdr_cfg, seed


def _run_task(
    sim_cfg: SimulatorConfig,
    est_cfg: EstimatorConfig,
    fdr_cfg: FDRConfig,
    seed: int,
) -> dict:
    res: DetectorResult = run_detector(sim_cfg, est_cfg, fdr_cfg, seed=seed)

    fdp_val = fdp(res.rejection_set, res.true_jumps)
    pw_val = power(res.rejection_set, res.true_jumps, k=2)
    f1_val = f1_score(res.rejection_set, res.true_jumps, k=2)
    delays = detection_delays(res.rejection_set, res.true_jumps, k=2)
    rr_val = rejection_rate(res.rejection_set, res.n_hypotheses)

    return {
        "algo": fdr_cfg.algo,
        "estimator": est_cfg.kind,
        "dt_seconds": sim_cfg.dt_seconds,
        "regime": sim_cfg.regime,
        "jump_size_sigma": sim_cfg.jump_size_sigma,
        "alpha": fdr_cfg.alpha,
        "seed": seed,
        "fdp": fdp_val,
        "power": pw_val,
        "f1": f1_val,
        "delay_mean": delays["mean"],
        "delay_median": delays["median"],
        "rej_rate": rr_val,
        "n_true_jumps": len(res.true_jumps),
        "n_rejections": len(res.rejection_set),
        "n_hypotheses": res.n_hypotheses,
        "wall_clock": res.wall_clock,
    }


# --------------------------------------------------------------------------- #
# Grid expansion                                                               #
# --------------------------------------------------------------------------- #


def _expand_grid(cfg: dict) -> list[tuple]:
    """Cartesian product of all grid dimensions, skipping inapplicable cells."""
    tasks = []
    for algo in cfg["algorithms"]:
        for estimator in cfg["estimators"]:
            for dt in cfg["frequencies_seconds"]:
                for regime in cfg["jump_regimes"]:
                    for jump_sz in cfg["jump_sizes_in_sigma"]:
                        for alpha in cfg["alpha"]:
                            # wealth_fractions only applies to e-LORD / e-SAFFRON
                            wf_list = cfg.get("wealth_fractions", [0.05])
                            if algo in ("elord", "esaffron"):
                                w1_vals = wf_list
                            else:
                                w1_vals = [0.05]  # ignored but keep code path clean
                            for w1 in w1_vals:
                                for seed in range(cfg["M_replications"]):
                                    tasks.append(
                                        _make_task(
                                            algo=algo,
                                            estimator=estimator,
                                            dt_seconds=float(dt),
                                            regime=regime,
                                            jump_size_sigma=float(jump_sz),
                                            alpha=float(alpha),
                                            w1=float(w1),
                                            seed=seed,
                                        )
                                    )
    return tasks


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def _print_ranking(df: pd.DataFrame, alpha: float) -> None:
    grp = df.groupby("algo")
    summary = grp.agg(
        fdr=("fdp", "mean"),
        fdr_se=("fdp", lambda x: x.std() / np.sqrt(len(x))),
        power=("power", "mean"),
        f1=("f1", "mean"),
        rej_rate=("rej_rate", "mean"),
        n_runs=("fdp", "count"),
    ).round(4)
    summary["fdr_criterion"] = summary["fdr"] <= (alpha + 2 * summary["fdr_se"])
    print(f"\n{'='*72}")
    print(f"Ranking table  (alpha={alpha}, criterion: FDR ≤ alpha + 2·SE)")
    print(f"{'='*72}")
    print(summary.sort_values("power", ascending=False).to_string())
    print()

    # Warn if any algo violates FDR control
    violators = summary[~summary["fdr_criterion"]].index.tolist()
    if violators:
        print(f"WARNING: FDR criterion VIOLATED for: {violators}")
    else:
        print("All algorithms satisfy FDR ≤ alpha + 2·SE  ✓")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo power grid for efdr-jumps")
    parser.add_argument(
        "--config", default="experiments/configs/grid_quick.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output parquet path (default: results/<config_stem>.parquet)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-2,
        help="joblib n_jobs (-2 = all cores but one, -1 = all cores)",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    output_path = Path(args.output) if args.output else (
        Path("results") / f"{cfg_path.stem}.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = _expand_grid(cfg)
    alpha_val = float(cfg["alpha"][0]) if cfg["alpha"] else 0.05

    print(f"Running {len(tasks)} tasks  (n_jobs={args.n_jobs})")
    t0 = time.perf_counter()

    rows = Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(_run_task)(sim, est, fdr, seed) for sim, est, fdr, seed in tasks
    )

    elapsed = time.perf_counter() - t0
    print(f"Completed in {elapsed:.1f}s  ({elapsed/len(tasks)*1000:.1f}ms/run)")

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"Results saved to {output_path}  ({len(df)} rows)")

    _print_ranking(df, alpha=alpha_val)


if __name__ == "__main__":
    main()
