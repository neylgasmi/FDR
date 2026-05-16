# CLAUDE.md — Online FDR Jump Detection Pipeline

> Project context file for Claude Code. This document is the source of truth for what to build, how to build it, and in what order. Keep it updated as the project evolves.

---

## 0. Project identity

**Name:** `efdr-jumps`
**One-liner:** Online FDR control for high-frequency jump detection in financial semimartingales, using e-processes built from jump-robust volatility estimators.
**Owner:** Neyl + binôme
**Stack:** Python 3.11+, numpy/scipy, numba for hot loops, pytest, matplotlib
**Deliverable:** Reproducible repo + research report (LaTeX, separate from this codebase)

---

## 1. What we are building (1-paragraph summary)

A research-grade Python library that (i) simulates intraday log-price paths under multiple jump-diffusion DGPs (Heston+Merton, Heston+Hawkes, optionally rough+jumps, optionally CGMY), (ii) computes jump-robust spot/integrated volatility estimators (MedRV, MinRV, bipower, threshold-bipower), (iii) constructs valid e-processes for the hypothesis "no jump at time i" from those estimators, (iv) runs state-of-the-art online FDR algorithms on the e-process stream (e-LOND, e-LORD, e-SAFFRON, e-GAI, stopped e-BH, online e-closure / compound e-BH), and (v) evaluates FDR-empirical and power across a calibrated experimental grid. Reproduces and extends Yen (2013) and Bajgrowicz-Scaillet (2016) using Ramdas-school 2025-2026 methodology.

---

## 2. Repository structure

```
efdr-jumps/
├── CLAUDE.md                          # this file
├── README.md                          # short, public-facing
├── pyproject.toml                     # uv / poetry / pip-compatible
├── .gitignore
├── .pre-commit-config.yaml            # ruff + black
│
├── src/efdr_jumps/
│   ├── __init__.py
│   ├── simulate/
│   │   ├── __init__.py
│   │   ├── heston.py                  # Heston SV (Andersen QE scheme)
│   │   ├── merton.py                  # Heston + Poisson-Merton jumps
│   │   ├── rough.py                   # (optional) rough Heston + jumps
│   │   ├── noise.py                   # microstructure noise (additive, one-sided)
│   │   └── base.py                    # PathSimulator ABC, ground-truth jump locations
│   │
│   ├── estimators/
│   │   ├── __init__.py
│   │   ├── bipower.py                 # BV, BNS test stat
│   │   ├── medrv_minrv.py             # Andersen-Dobrev-Schaumburg 2012
│   │   ├── threshold.py               # Mancini truncation
│   │   ├── preavg.py                  # pre-averaging (Bibinger), noise-robust
│   │   └── spot.py                    # local windowed spot volatility
│   │
│   ├── evalues/
│   │   ├── __init__.py
│   │   ├── construct.py               # E_i from r_i and sigma_hat_i
│   │   ├── mixture.py                 # mixture e-values, Robbins-style
│   │   └── eprocess.py                # sequential e-processes (local/global)
│   │
│   ├── fdr/
│   │   ├── __init__.py
│   │   ├── ebh.py                     # offline e-BH (Wang-Ramdas 2022)
│   │   ├── stopped_ebh.py             # stopped e-BH (Wang-Dandapanthula-Ramdas 2025)
│   │   ├── elond.py                   # e-LOND (Xu-Ramdas 2024)
│   │   ├── elord_esaffron.py          # e-LORD, e-SAFFRON (e-GAI, Zhang et al. 2025)
│   │   ├── eclosure.py                # online e-closure (Xu-Fischer-Ramdas 2026)
│   │   ├── compound_ebh.py            # compound e-BH with donation (XFR 2026)
│   │   └── baselines.py               # BH/BY on Lee-Mykland and BNS p-values
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── detector.py                # full online pipeline orchestrator
│   │   └── metrics.py                 # FDR/power/F1/detection-delay
│   │
│   └── utils/
│       ├── __init__.py
│       ├── stats.py                   # quantile interpolation, Gumbel, etc.
│       └── viz.py                     # plotting helpers
│
├── tests/
│   ├── test_simulate.py
│   ├── test_estimators.py
│   ├── test_evalues_validity.py       # empirical check that E[E_i] <= 1 under H0
│   ├── test_fdr_algorithms.py         # FDR is controlled under H0-only streams
│   └── test_pipeline.py
│
├── experiments/
│   ├── 01_evalue_validity.py          # under H0 only, simulate and check FDR<=alpha
│   ├── 02_power_grid.py               # full grid simulation (M=1000 reps)
│   ├── 03_baseline_comparison.py      # vs BH-offline (Yen 2013, BS 2016)
│   ├── 04_real_data.py                # TAQ/LOBSTER/BMLL application
│   └── configs/
│       ├── grid_main.yaml
│       └── grid_quick.yaml            # reduced grid for dev iterations
│
├── notebooks/
│   ├── 00_simulation_sanity_checks.ipynb
│   ├── 01_estimator_comparison.ipynb
│   ├── 02_evalue_distribution_under_H0.ipynb
│   └── 03_results_figures.ipynb       # generates final paper figures
│
└── data/
    ├── raw/                           # gitignored
    ├── processed/                     # gitignored
    └── README.md                      # source attribution
```

---

## 3. Build order (sequential, each step has an exit criterion)

Claude Code should implement modules in this order. Don't skip ahead; each layer is consumed by the next.

### Step 1 — Repo bootstrap

- `pyproject.toml` with deps: numpy, scipy, numba, matplotlib, pandas, pytest, pytest-cov, ruff, black, pyyaml.
- `pre-commit` hooks: ruff, black, no large files.
- Empty package skeleton; `pytest` runs (with zero tests) and exits 0.
- GitHub Actions: lint + test on push.
- **Exit criterion:** `pip install -e .` works, `pytest` passes, `ruff check .` passes.

### Step 2 — Simulation primitives (`src/efdr_jumps/simulate/`)

Implement in this sub-order:

1. **`base.py`** — `PathSimulator` ABC returning `(times, log_price, jump_indices, sigma_path)`. Ground truth jump locations must be exposed for evaluation.
2. **`heston.py`** — Heston model with Andersen QE scheme (no jumps yet). Use as baseline.
3. **`merton.py`** — extends Heston with Poisson-Merton jumps. Reference: Andersen-Benzoni-Lund 2002 parameterization.
4. **`noise.py`** — additive gaussian microstructure noise and (optional) one-sided noise à la Bibinger 2024.
5. **`rough.py`** *(optional, only after Steps 4-7 below work)* — rough Heston via hybrid scheme (Bayer-Friz-Gatheral) with additive jumps.

**Tests required:** 
- Each simulator passes a Kolmogorov-Smirnov sanity check on increments under no-jump regime.
- Jump indices are exactly recoverable from the path.

**Exit criterion:** simulate 1 day of 1-second prices under Heston-Merton in < 100ms.

### Step 3 — Estimators (`src/efdr_jumps/estimators/`)

1. **`bipower.py`** — BV, BNS jump test statistic, Lee-Mykland statistic.
2. **`medrv_minrv.py`** — Andersen-Dobrev-Schaumburg 2012, including scale factor corrections.
3. **`threshold.py`** — Mancini 2009 truncated power variation.
4. **`spot.py`** — windowed spot volatility from each of the above, with configurable window K. **Critical:** the window used to estimate σ̂_i must *exclude* the test point i to preserve e-value validity.
5. **`preavg.py`** — pre-averaging à la Jacod-Li-Mykland-Podolskij-Vetter for noise-robust versions.

**Tests required:**
- All estimators converge to true integrated variance under no-jump diffusion as Δ→0 (verified on M=200 reps at increasing frequencies).
- Bias under jumps: MedRV/MinRV should have less than 10% bias at jump intensity 5/day; BV without truncation should be visibly biased (used as negative control).

**Exit criterion:** estimator comparison table reproduces ADS 2012 Table 1 within reasonable Monte Carlo error.

### Step 4 — E-values (`src/efdr_jumps/evalues/`)

This is the **core theoretical contribution** of the project. Implementations need to be cross-checked against §3.1 of the written report (Problème 1).

1. **`construct.py`** — given (r_i, σ̂_i), build E_i. Initial baseline: ratio of likelihoods between N(0, Δσ̂²) and a mixture alternative N(μ, Δσ̂² + σ_J²) integrated over μ ~ prior. Document explicitly the mixing distribution.
2. **`mixture.py`** — Robbins-style mixture e-values (composite null/alternative).
3. **`eprocess.py`** — sequential e-process construction with explicit filtration handling. Must distinguish *local* (built within the test point's window) and *global* (consistent with the master filtration) per Wang-Dandapanthula-Ramdas 2025.

**Tests required (rationale spelled out — these checks together establish that our e-values are statistically meaningful, not just legally valid):**

- **Distribution of E_i under H0 vs H1.** Simulate M=10000 paths under pure diffusion (H0) and M=10000 paths with a saut of fixed size α·σ injected at a known location. Compare the empirical distributions of E_i. Under H0, mass should concentrate near or below 1; under H1, mass should shift to large values. *Why:* this is the most direct visual diagnostic that the e-value separates null from alternative — without separation, no FDR algorithm downstream can produce power.

- **E-power across alternative regimes.** Compute the *e-power* defined as $\mathbb{E}_{H_1}[\log E_i]$, which is the proper notion of "power" for e-values (Grünwald, de Heide, Koolen 2024; Ramdas-Wang 2025). E-power must be (a) strictly positive at the alternative, (b) monotone increasing in jump size, (c) reasonably stable across volatility regimes. *Why:* unlike standard power (probability of rejection), e-power is the right efficiency criterion for sequential testing — it directly governs how fast wealth accumulates in α-investing procedures. A test with high e-power will dominate in any downstream online FDR procedure.

- **Significance against trivial baselines.** Compare E_i against (i) the trivial e-value E ≡ 1 (always-valid, zero-power floor) and (ii) the calibrated Lee-Mykland p-value converted to a p-to-e calibrator (Vovk-Wang 2021). E_i must dominate both in e-power for the construction to be worth using. *Why:* if our e-value is not better than a calibrated p-value, we haven't gained anything from going to the e-value framework — this test is a sanity check that the construction is actually exploiting the structure of the problem.

- **Validity check as a side product.** Under pure H0 over M=10000 reps, the empirical mean of E_i over the conditional filtration must be ≤ 1 + 3·SE. This is a *necessary* condition for FDR control downstream, not the headline result. Folded into the construction tests rather than its own module.

**Exit criterion:** `experiments/01_evalue_validity.py` runs and produces (a) the H0-vs-H1 distribution figure with clear separation, (b) the e-power curve as a function of jump size, (c) the comparison against the p-to-e calibrator baseline.

### Step 5 — FDR algorithms (`src/efdr_jumps/fdr/`)

Each implements a function with signature:

```python
def online_fdr(e_values: Iterator[float], alpha: float, **kwargs) -> Iterator[bool]:
    """Yields rejection decisions as e_values stream in."""
```

Implement in this order:

1. **`ebh.py`** — offline e-BH (Wang & Ramdas 2022). Sanity benchmark.
2. **`elond.py`** — e-LOND (Xu & Ramdas 2024). First online baseline.
3. **`elord_esaffron.py`** — e-LORD and e-SAFFRON from e-GAI framework (Zhang, Wei, Ren, Zou 2025, arXiv:2506.01452).
4. **`stopped_ebh.py`** — stopped e-BH (Wang, Dandapanthula, Ramdas 2025, arXiv:2502.08539). **Read the paper before implementing — the local-vs-global e-process distinction matters here.**
5. **`eclosure.py`** — online e-closure (Xu, Fischer, Ramdas 2026, arXiv:2603.24792). Must hit O(log t) per step.
6. **`compound_ebh.py`** — online compound e-BH with donation (same paper).
7. **`baselines.py`** — BH and BY applied to Lee-Mykland and BNS p-values, offline. Reproduces Yen 2013 and Bajgrowicz-Scaillet 2016 as baselines.

**Tests required (each algorithm is tested in isolation for correctness, then compared head-to-head):**

- **Self-validity per algorithm.** For each algorithm, simulate a stream of pure-H0 e-values (independent Pareto-1 or constant 1, plus our actual e-values under no-jump diffusion) and verify empirical FDR ≤ α + MC tolerance on M=1000 reps. *Why:* each procedure has its own dependence assumptions (e-LOND under arbitrary dependence, stopped e-BH under the local-global causal condition, etc.) — we need to check that the assumption is actually met by our e-value stream before any cross-comparison is meaningful.

- **Deterministic stream cross-check.** Build a hand-crafted finite stream where the rejection set is mathematically pre-computable (e.g., five e-values at 1, 1, 20, 1, 1 with α=0.1). All six algorithms should agree on this stream — if they don't, the implementation is wrong. *Why:* differences in rejection sets on a noise-free input isolate implementation bugs from genuine algorithmic differences.

- **Head-to-head comparison once calibrated — supervised regime.** Once every algorithm is verified to control FDR at the target α on H0-only streams, simulate paths *with known jump times* (oracle ground truth) under Heston+Merton at jump intensities {rare, moderate, dense}. Compare on: empirical FDR (must remain ≤ α), power, detection delay, F1 on localization. *We do not expect the algorithms to agree on rejection sets — that is the point.* Different procedures explore the alpha-wealth budget differently (LOND spends conservatively, SAFFRON adapts to past rejections, stopped e-BH is anytime-valid which costs power, e-closure tightens via compound construction). The objective is to map *which procedure dominates in which regime*. *Why:* this is the supervised benchmark — we have ground truth, so we can measure absolute performance.

- **Head-to-head comparison — unsupervised regime.** On real intraday data where the true jump set is unknown, compare the algorithms by (a) agreement matrix between procedures (Jaccard on rejection sets), (b) stability under sub-sampling, (c) economic plausibility (alignment of detections with scheduled macro announcements as in Bajgrowicz-Scaillet 2016). *Why:* in practice we never know the true jumps. An algorithm that looks great under supervised simulation but produces unstable or implausible detections on real data is not deployable. This unsupervised benchmark tells us which procedure is *credible*, not just *correct*.

- **Power monotonicity sanity.** Within methods sharing the same theoretical guarantees, more powerful procedures (e-closure improvements over e-LOND, compound e-BH improvements over e-BH) should make at least as many rejections on the same stream in expectation. Verify this on M=200 reps. *Why:* a theoretical "strict improvement" result must hold empirically in the simulation, otherwise either the implementation is wrong or the regime is pathological.

**Exit criterion:** all six algorithms pass self-validity and deterministic cross-check; the supervised comparison produces a clear ranking table for each (frequency × jump regime × α) cell; the unsupervised comparison produces an agreement matrix on at least one real data sample.

### Step 6 — Pipeline orchestrator (`src/efdr_jumps/pipeline/`)

`detector.py` glues everything: takes a `PathSimulator` config + estimator choice + FDR algorithm, returns rejection set + metrics.

`metrics.py` computes:
- Empirical FDR (averaged over M reps)
- Power
- Detection delay (mean time between true jump and rejection)
- F1 on jump localization (with tolerance window of ±k ticks)
- Wall-clock per algorithm

**Exit criterion:** end-to-end run in `experiments/02_power_grid.py` with `grid_quick.yaml` completes in < 5 minutes.

### Step 7 — Experiments

`experiments/02_power_grid.py` runs the full Monte Carlo grid defined in `configs/grid_main.yaml`:

```yaml
frequencies_seconds: [1, 5, 30, 60, 300]
jump_regimes: [rare, moderate, hawkes_dense]
jump_sizes_in_sigma: [3, 5, mixed]
alpha: [0.05, 0.10]
wealth_fractions: [0.1, 0.25, 0.5, 0.75]
M_replications: 1000
algorithms: [bh_lm, bh_bns, elond, elord, esaffron, egai, stopped_ebh, eclosure, compound_ebh]
estimators: [bv, medrv, minrv, threshold]
```

The `wealth_fractions` parameter governs how much of the available α-wealth each rejection consumes in the GAI-family procedures (e-LORD, e-SAFFRON, e-GAI). Smaller fractions are more conservative (lower α-death risk, lower instantaneous power); larger fractions are aggressive (faster wealth depletion, higher per-rejection sensitivity). The original e-GAI paper introduces *risk-averse* allocation as a default but does not optimize over it — we treat it as a tunable parameter and study its effect on the FDR-power tradeoff. *Why this matters:* in a streaming jump-detection setting, the α-death phenomenon (running out of wealth before the day ends) is a real risk for procedures like LORD/SAFFRON. Studying the wealth-fraction × regime interaction tells us how to size the parameter for actual intraday deployment.

For procedures where the wealth parameter is not applicable (e-LOND, stopped e-BH, e-closure, offline baselines), the parameter is ignored. The grid is therefore not a full Cartesian product — encode the inapplicable cells as skipped in the runner.

Should be embarrassingly parallel; use `multiprocessing.Pool` or `joblib`.

Generates a single results parquet file consumed by `notebooks/03_results_figures.ipynb`.

### Step 8 — Real data application (`experiments/04_real_data.py`)

**Only after Step 7 is solid.** Inputs: cleaned intraday data for a basket of stocks. Plan A: BMLL Level 3 (if available). Plan B: a TAQ sample for ~10 large caps over one month. Plan C: LOBSTER public sample.

**Required input granularity:**

- **Minimum frequency:** 1-second mid-quote or trade prices. Anything coarser than 1-minute defeats the purpose of an online procedure designed for HF — fall back to Plan C rather than running on daily/5-min data.
- **Native granularity preferred:** tick-by-tick (event time), then downsampled in code to a regular grid (calendar time) at the configured `frequencies_seconds`. Do not accept pre-aggregated bars from the data vendor — we want control over the aggregation step.
- **Required fields per record:** timestamp (microsecond precision or better), price (trade) or mid-quote (best bid + best ask / 2 for LOB sources), volume, exchange/venue code if available.
- **Trading hours filter:** market open + 5 minutes to market close - 5 minutes (excludes opening auction noise and closing-cross volatility), per Bajgrowicz-Scaillet 2016 convention.
- **Cleaning required upstream of the pipeline:** zero-price filter, out-of-sequence timestamp filter, bounce-back outlier filter (Brownlees-Gallo 2006), median aggregation when multiple ticks share a timestamp.
- **For Plan A (BMLL Level 3):** full LOB snapshots with best-bid and best-ask levels enable the one-sided noise model of Bibinger 2024 — that's the regime where our setup has the strongest theoretical edge. If BMLL is available, use best-ask quote stream and the one-sided noise estimator, not the additive MMN model.
- **For Plan B (TAQ):** trade prices with millisecond timestamps. Use the standard additive MMN model. Subsample to 5-second grid for the baseline run.
- **For Plan C (LOBSTER):** message-level LOB data is available for ~10 stocks. Re-construct best-bid/best-ask from messages and treat like Plan A.
- **Sample size:** at minimum one full trading day for one asset (smoke test), realistically one month × 5-10 assets for the report's empirical section.

Output: number of detected jumps per stock per day per algorithm, agreement matrix between procedures, alignment with FOMC / earnings / macro announcement timestamps (cross-checked against a news calendar).

---

## 4. Conventions

### Code style

- Type hints everywhere (`from __future__ import annotations` at the top).
- Docstrings in NumPy style. Each public function references the paper + equation/algorithm it implements.
- No comments saying *what* the code does; comments only when explaining *why* (e.g., "scale factor from ADS 2012 eq. 7").
- Hot loops vectorized with numpy; if a Python loop is genuinely needed, decorate with `@numba.njit`.
- No magic numbers — every constant is a named kwarg with default.

### Testing

- pytest with `-q --tb=short`.
- Slow tests (>1s) marked `@pytest.mark.slow` and excluded from default run.
- Use `numpy.testing.assert_allclose` for numerical comparisons; never `==` on floats.
- Random tests use a fixed seed; document the seed in the test.

### Numerical conventions

- Time always in seconds since market open.
- Log-prices always natural log.
- All volatilities are *spot* volatilities in annualized units unless suffixed `_daily` or `_intraday`.
- Δ = sampling interval in years (so 1 second = 1/(252·6.5·3600)).
- α (FDR target) is always in (0, 1), never in percent.

### Git hygiene

- Branch per feature: `step-N-description`.
- Commit messages in imperative ("Add MedRV estimator", not "Added").
- PRs reference the step in this document.
- No data files in git (everything in `data/` is gitignored except `data/README.md`).

---

## 5. Acceptance criteria for the whole project

1. **Repo passes CI**: lint + tests on every push.
2. **Validity holds**: under H0-only simulations, empirical FDR ≤ α + 2·SE for every algorithm × every estimator combination at every frequency.
3. **Power gain documented**: online e-closure (XFR 2026) shows ≥ 10% power gain over e-LOND at fixed FDR=0.10, jump regime moderate, frequency 5s.
4. **Reproduces literature**: BH on Lee-Mykland p-values reproduces Yen 2013 Table 2 within MC error.
5. **All figures regenerable**: running `notebooks/03_results_figures.ipynb` end-to-end produces every figure in the written report.
6. **Real data smoke test**: at least one day of one asset processed end-to-end.

---

## 6. What is out of scope

Explicitly **not** doing in this project (to keep scope bounded):

- Multi-asset cojump detection (Jacod-Todorov 2009 framework) — mentioned in the report's discussion section only.
- Trading strategies built on jump signals — pure inference project.
- Bayesian methods (BOCPD etc.) — frequentist FDR only.
- Deep learning detectors — outside research question.
- Web UI / dashboard — CLI scripts + notebooks are enough.

---

## 7. Quick commands

```bash
# install
pip install -e ".[dev]"

# lint + format
ruff check . && black .

# tests (fast only)
pytest -q

# tests (all, including slow MC)
pytest -q -m "not skip" --runslow

# quick grid (for dev)
python experiments/02_power_grid.py --config experiments/configs/grid_quick.yaml

# full grid (overnight)
python experiments/02_power_grid.py --config experiments/configs/grid_main.yaml

# regenerate paper figures
jupyter nbconvert --to notebook --execute notebooks/03_results_figures.ipynb
```

---

## 8. Reading prerequisites before implementing

Claude Code should consult these papers when implementing the corresponding module:

| Module | Paper | arXiv |
|---|---|---|
| `estimators/medrv_minrv.py` | Andersen, Dobrev, Schaumburg (2012) | JoE |
| `estimators/threshold.py` | Mancini (2009) | Scand. J. Statist. |
| `evalues/eprocess.py` | Ramdas, Ruf, Larsson, Koolen (2022) | 2009.03167 |
| `evalues/construct.py` | Grünwald, de Heide, Koolen (2024) — e-power | Annals of Statistics |
| `fdr/stopped_ebh.py` | Wang, Dandapanthula, Ramdas (2025) | 2502.08539 |
| `fdr/eclosure.py` | Xu, Fischer, Ramdas (2026) | 2603.24792 |
| `fdr/elord_esaffron.py` | Zhang, Wei, Ren, Zou (2025) e-GAI | 2506.01452 |
| `fdr/baselines.py` | Yen (2013), Bajgrowicz-Scaillet (2016) | PLOS One / MS |

Don't write code for a module until the corresponding paper has been read.

---

## 9. Workflow rule for Claude Code sessions

For any non-trivial change:

1. State the step from §3 you're about to work on.
2. List the files you'll touch.
3. Run `pytest` on the affected module after implementation, before moving on.
4. Update this CLAUDE.md if conventions or scope shift.
5. Don't write the LaTeX report — that lives in a separate repo.

---

*Last updated: 15 May 2026 — revised after binôme review (Step 2: dropped Hawkes; Step 4: dropped standalone validity module, sharpened tests around e-power; Step 5: reframed comparison around calibrated supervised vs unsupervised regimes; Step 7: added wealth-fraction parameter; Step 8: specified data granularity).*
