"""
03_results_figures.py — Figures pour efdr-jumps grid_main results.

Usage:
    python notebooks/03_results_figures.py
    python notebooks/03_results_figures.py \\
        --input results/grid_main.parquet --output results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# config
ALGO_ORDER = ["bh_lm", "bh_bns", "ebh", "stopped_ebh", "elond", "elord", "esaffron"]
ALGO_LABELS = {
    "bh_lm": "BH-LM",
    "bh_bns": "BH-BNS",
    "ebh": "e-BH",
    "stopped_ebh": "Stopped e-BH",
    "elond": "e-LOND",
    "elord": "e-LORD",
    "esaffron": "e-SAFFRON",
}
ALGO_COLORS = {
    "bh_lm": "#e63946",
    "bh_bns": "#f4a261",
    "ebh": "#2a9d8f",
    "stopped_ebh": "#264653",
    "elond": "#457b9d",
    "elord": "#6a4c93",
    "esaffron": "#a8dadc",
}
REGIME_LABELS = {"rare": "Rare jumps", "moderate": "Moderate jumps", "hawkes_dense": "Hawkes dense"}
DT_LABELS = {1: "1s", 5: "5s", 30: "30s", 60: "1min", 300: "5min"}

plt.rcParams.update(
    {
        "font.family": "monospace",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 150,
    }
)


# helpers
def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # For elord/esaffron, aggregate over wealth_fractions (best w1)
    df = df.groupby(
        ["algo", "estimator", "dt_seconds", "regime", "jump_size_sigma", "alpha", "seed"],
        as_index=False,
    ).agg(
        fdp=("fdp", "mean"),
        power=("power", "mean"),
        f1=("f1", "mean"),
        n_true_jumps=("n_true_jumps", "mean"),
        n_rejections=("n_rejections", "mean"),
    )
    return df


def agg(df: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    g = df.groupby(groupby)
    out = g.agg(
        fdr=("fdp", "mean"),
        fdr_se=("fdp", lambda x: x.std() / np.sqrt(len(x))),
        power=("power", "mean"),
        power_se=("power", lambda x: x.std() / np.sqrt(len(x))),
        f1=("f1", "mean"),
        n=("fdp", "count"),
    ).reset_index()
    return out


# figure 1 — fdr vs frequency, per regime
def fig_fdr_vs_freq(df: pd.DataFrame, alpha: float, output_dir: Path) -> None:
    regimes = ["rare", "moderate", "hawkes_dense"]
    algos = [a for a in ALGO_ORDER if a in df["algo"].unique()]
    sub = df[df["alpha"] == alpha]
    agged = agg(sub, ["algo", "dt_seconds", "regime"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(f"FDR vs Sampling Frequency  (α={alpha})", fontsize=13, fontweight="bold", y=1.01)

    for ax, regime in zip(axes, regimes):
        r = agged[agged["regime"] == regime]
        dts = sorted(r["dt_seconds"].unique())
        x = np.arange(len(dts))

        for algo in algos:
            a = r[r["algo"] == algo].set_index("dt_seconds")
            y = [a.loc[dt, "fdr"] if dt in a.index else np.nan for dt in dts]
            ye = [a.loc[dt, "fdr_se"] if dt in a.index else np.nan for dt in dts]
            ax.plot(
                x,
                y,
                marker="o",
                label=ALGO_LABELS[algo],
                color=ALGO_COLORS[algo],
                linewidth=1.8,
                markersize=5,
            )
            ax.fill_between(
                x,
                [v - 2 * e if v == v else np.nan for v, e in zip(y, ye)],
                [v + 2 * e if v == v else np.nan for v, e in zip(y, ye)],
                alpha=0.12,
                color=ALGO_COLORS[algo],
            )

        ax.axhline(alpha, color="black", linestyle=":", linewidth=1.2, label=f"α={alpha}")
        ax.set_xticks(x)
        ax.set_xticklabels([DT_LABELS.get(int(d), str(d)) for d in dts])
        ax.set_title(REGIME_LABELS.get(regime, regime), fontweight="bold")
        ax.set_xlabel("Sampling frequency")
        if ax == axes[0]:
            ax.set_ylabel("FDR")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    out = output_dir / f"fig1_fdr_vs_freq_alpha{int(alpha*100)}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# figure 2 — power vs frequency, per regime
def fig_power_vs_freq(df: pd.DataFrame, alpha: float, output_dir: Path) -> None:
    regimes = ["rare", "moderate", "hawkes_dense"]
    algos = [a for a in ALGO_ORDER if a in df["algo"].unique()]
    sub = df[df["alpha"] == alpha]
    agged = agg(sub, ["algo", "dt_seconds", "regime"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(
        f"Power vs Sampling Frequency  (α={alpha})", fontsize=13, fontweight="bold", y=1.01
    )

    for ax, regime in zip(axes, regimes):
        r = agged[agged["regime"] == regime]
        dts = sorted(r["dt_seconds"].unique())
        x = np.arange(len(dts))

        for algo in algos:
            a = r[r["algo"] == algo].set_index("dt_seconds")
            y = [a.loc[dt, "power"] if dt in a.index else np.nan for dt in dts]
            ye = [a.loc[dt, "power_se"] if dt in a.index else np.nan for dt in dts]
            ax.plot(
                x,
                y,
                marker="o",
                label=ALGO_LABELS[algo],
                color=ALGO_COLORS[algo],
                linewidth=1.8,
                markersize=5,
            )
            ax.fill_between(
                x,
                [v - 2 * e if v == v else np.nan for v, e in zip(y, ye)],
                [v + 2 * e if v == v else np.nan for v, e in zip(y, ye)],
                alpha=0.12,
                color=ALGO_COLORS[algo],
            )

        ax.set_xticks(x)
        ax.set_xticklabels([DT_LABELS.get(int(d), str(d)) for d in dts])
        ax.set_title(REGIME_LABELS.get(regime, regime), fontweight="bold")
        ax.set_xlabel("Sampling frequency")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        if ax == axes[0]:
            ax.set_ylabel("Power")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    out = output_dir / f"fig2_power_vs_freq_alpha{int(alpha*100)}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# figure 3 — fdr/power tradeoff scatter
def fig_tradeoff(df: pd.DataFrame, alpha: float, output_dir: Path) -> None:
    algos = [a for a in ALGO_ORDER if a in df["algo"].unique()]
    sub = df[df["alpha"] == alpha]
    agged = agg(sub, ["algo", "regime"])

    regimes = ["rare", "moderate", "hawkes_dense"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(f"FDR–Power Tradeoff  (α={alpha})", fontsize=13, fontweight="bold", y=1.01)

    for ax, regime in zip(axes, regimes):
        r = agged[agged["regime"] == regime]
        for algo in algos:
            a = r[r["algo"] == algo]
            if a.empty:
                continue
            ax.scatter(
                a["fdr"].values[0],
                a["power"].values[0],
                color=ALGO_COLORS[algo],
                s=120,
                zorder=5,
                label=ALGO_LABELS[algo],
                edgecolors="white",
                linewidth=0.8,
            )
            ax.annotate(
                ALGO_LABELS[algo],
                (a["fdr"].values[0], a["power"].values[0]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
            )

        ax.axvline(alpha, color="black", linestyle=":", linewidth=1.2, label=f"α={alpha}")
        ax.set_title(REGIME_LABELS.get(regime, regime), fontweight="bold")
        ax.set_xlabel("FDR")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        if ax == axes[0]:
            ax.set_ylabel("Power")

    fig.tight_layout()
    out = output_dir / f"fig3_tradeoff_alpha{int(alpha*100)}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# figure 4 — ranking table heatmap
def fig_heatmap(df: pd.DataFrame, alpha: float, output_dir: Path) -> None:
    algos = [a for a in ALGO_ORDER if a in df["algo"].unique()]
    sub = df[df["alpha"] == alpha]
    agged = agg(sub, ["algo"])
    agged = agged.set_index("algo")
    agged = agged.reindex([a for a in algos if a in agged.index])
    agged = agged.dropna(how="all")

    if agged.empty:
        print("  Skipping fig4: no data.")
        return

    metrics = ["fdr", "power", "f1"]
    metric_labels = ["FDR", "Power", "F1"]
    data = agged[metrics].values

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data.T, aspect="auto", cmap="RdYlGn_r")
    if data.size > 0 and data.max() > 0:
        im.set_clim(0, data.max())

    ax.set_xticks(range(len(agged)))
    ax.set_xticklabels([ALGO_LABELS.get(a, a) for a in agged.index], rotation=30, ha="right")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metric_labels)

    for i, algo in enumerate(agged.index):
        for j, metric in enumerate(metrics):
            val = data[i, j]
            ax.text(
                i,
                j,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black" if val < data.max() * 0.7 else "white",
            )

    ax.set_title(f"Algorithm Ranking Heatmap  (α={alpha})", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = output_dir / f"fig4_heatmap_alpha{int(alpha*100)}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# figure 5 — effect of jump size
def fig_jump_size(df: pd.DataFrame, alpha: float, output_dir: Path) -> None:
    algos = [a for a in ALGO_ORDER if a in df["algo"].unique()]
    sub = df[df["alpha"] == alpha]
    jump_sizes = sorted(sub["jump_size_sigma"].unique())

    if len(jump_sizes) < 2:
        print("  Skipping fig5: only one jump size in data.")
        return

    agged = agg(sub, ["algo", "jump_size_sigma"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Effect of Jump Size  (α={alpha})", fontsize=13, fontweight="bold", y=1.01)

    for ax, metric, label in zip(axes, ["fdr", "power"], ["FDR", "Power"]):
        x = np.arange(len(jump_sizes))
        width = 0.8 / len(algos)
        for k, algo in enumerate(algos):
            a = agged[agged["algo"] == algo].set_index("jump_size_sigma")
            y = [a.loc[js, metric] if js in a.index else np.nan for js in jump_sizes]
            offset = (k - len(algos) / 2 + 0.5) * width
            ax.bar(
                x + offset, y, width=width * 0.9, color=ALGO_COLORS[algo], label=ALGO_LABELS[algo]
            )

        if metric == "fdr":
            ax.axhline(alpha, color="black", linestyle=":", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{js}σ" for js in jump_sizes])
        ax.set_xlabel("Jump size")
        ax.set_ylabel(label)
        ax.set_title(label)
        if metric == "power":
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    out = output_dir / f"fig5_jump_size_alpha{int(alpha*100)}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/grid_main.parquet")
    parser.add_argument("--output", default="results/figures")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input}...")
    df = load(args.input)
    print(f"  {len(df)} rows, algos: {sorted(df['algo'].unique())}")

    for alpha in sorted(df["alpha"].unique()):
        print(f"\n--- alpha={alpha} ---")
        fig_fdr_vs_freq(df, alpha, output_dir)
        fig_power_vs_freq(df, alpha, output_dir)
        fig_tradeoff(df, alpha, output_dir)
        fig_heatmap(df, alpha, output_dir)
        fig_jump_size(df, alpha, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
