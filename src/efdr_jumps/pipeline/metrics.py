from __future__ import annotations

import numpy as np


def fdp(rejection_set: frozenset[int], true_jumps: frozenset[int]) -> float:
    """False Discovery Proportion for a single run (no tolerance).

    FDP = |R \\ J| / max(|R|, 1)   where J = true jump indices.
    """
    if not rejection_set:
        return 0.0
    fp = len(rejection_set - true_jumps)
    return fp / len(rejection_set)


def power(
    rejection_set: frozenset[int],
    true_jumps: frozenset[int],
    k: int = 2,
) -> float:
    """Fraction of true jumps with at least one rejection within ±k ticks.

    Returns NaN when there are no true jumps (undefined).
    """
    if not true_jumps:
        return float("nan")
    detected = sum(any((t + d) in rejection_set for d in range(-k, k + 1)) for t in true_jumps)
    return detected / len(true_jumps)


def f1_score(
    rejection_set: frozenset[int],
    true_jumps: frozenset[int],
    k: int = 2,
) -> float:
    """F1 on jump localization with ±k tick tolerance.

    Precision = fraction of rejections within k of at least one true jump.
    Recall    = fraction of true jumps with at least one rejection within k.
    """
    if not true_jumps and not rejection_set:
        return 1.0
    if not true_jumps or not rejection_set:
        return 0.0

    prec = sum(any((r + d) in true_jumps for d in range(-k, k + 1)) for r in rejection_set) / len(
        rejection_set
    )

    rec = power(rejection_set, true_jumps, k)

    if prec + rec == 0.0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def detection_delays(
    rejection_set: frozenset[int],
    true_jumps: frozenset[int],
    k: int = 2,
) -> dict[str, float]:
    """Mean and median tick distance from each true jump to its nearest detection.

    Only counts true jumps that were detected within ±k (i.e., have a hit).
    Returns NaN keys when no true jump was detected.
    """
    delays = []
    for t in true_jumps:
        hits = [abs(r - t) for r in rejection_set if abs(r - t) <= k]
        if hits:
            delays.append(min(hits))

    if not delays:
        return {"mean": float("nan"), "median": float("nan")}
    return {"mean": float(np.mean(delays)), "median": float(np.median(delays))}


def rejection_rate(rejection_set: frozenset[int], n_hypotheses: int) -> float:
    """Fraction of hypotheses rejected: |R| / n."""
    if n_hypotheses <= 0:
        return 0.0
    return len(rejection_set) / n_hypotheses


def aggregate(
    results: list[dict[str, float]],
) -> dict[str, float]:
    """Aggregate per-run metric dicts over M replications.

    Each dict must have keys: fdp, power, f1, delay_mean, delay_median, rej_rate.
    NaN values are excluded from power/F1/delay means.
    Returns: mean and SE for each scalar metric.
    """
    keys = ["fdp", "power", "f1", "delay_mean", "delay_median", "rej_rate"]
    out: dict[str, float] = {}
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=float)
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            out[k] = float("nan")
            out[f"{k}_se"] = float("nan")
        else:
            out[k] = float(np.mean(valid))
            out[f"{k}_se"] = float(np.std(valid) / np.sqrt(len(valid)))
    return out
