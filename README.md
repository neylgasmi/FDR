# efdr-jumps

**E-values for sequential jump detection in high-frequency with online FDR guarantees**

M1 MAEF research project : Neyl Gasmi & Jules Brion  
Supervisor: Alain Celisse (SAMM, Université Paris 1) : 2025-2026

---

## What this is

This project detects discontinuities (jumps) in high-frequency financial data in real time while controlling the false discovery rate (FDR) sequentially. The core finding: classical BH multiple testing violates FDR guarantees at 5-second frequency, while e-value-based procedures control it under arbitrary dependence.

**Paper:** [`paper/publish.pdf`](paper/publish.pdf)

---

## Results at a glance

At Δt = 5s, α = 0.05, M = 100 Monte-Carlo replications:

| Algorithm | FDR | Power | FDR controlled? |
|---|---|---|---|
| BH-LM / BH-BNS | 0.096 | 0.441 | No (threshold: 0.094) |
| Stopped e-BH | 0.016 | 0.349 | Yes |
| e-BH | 0.010 | 0.326 | Yes |
| e-LOND | 0.001 | 0.235 | Yes |

---

## Project structure

```
efdr-jumps/
├── src/efdr_jumps/       # Python package (5-layer pipeline)
│   ├── simulate/         # Heston + Merton + Hawkes processes
│   ├── estimators/       # BV, MedRV, MinRV spot volatility
│   ├── evalues/          # GROW e-value construction
│   ├── fdr/              # 7 FDR procedures
│   └── pipeline/         # Monte-Carlo runner
├── tests/                # 86 unit tests
├── experiments/          # Grid configs + runner
│   └── configs/          # grid_quick / grid_medium / grid_main
├── notebooks/            # Walkthrough + result figures
├── results/figures/      # Monte-Carlo output figures
├── paper/                # LaTeX source + compiled PDF
└── docs/                 # Technical decisions (ADRs)
```

---

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.10.

---

## Run tests

```bash
pytest tests/
```

---

## Reproduce the Monte-Carlo grid

```bash
python experiments/02_power_grid.py --config experiments/configs/grid_medium.yaml
```

`grid_medium`: 100 replications × 500 steps × 3 frequencies × 3 regimes ≈ 7 min.

---

## Procedures implemented

| Procedure | Type | Dependence guarantee | Reference |
|---|---|---|---|
| e-BH | Offline | Arbitrary | Wang & Ramdas 2022 |
| e-LOND | Online | Arbitrary | Xu & Ramdas 2024 |
| e-LORD | Online | Arbitrary | Zhang et al. 2025 |
| e-SAFFRON | Online | Arbitrary | Zhang et al. 2025 |
| Stopped e-BH | Online (anytime) | Markov / causal | Wang et al. 2025 |
| BH-LM | Offline | PRDS | Lee & Mykland 2008 |
| BH-BNS | Offline | PRDS | Bajgrowicz & Scaillet 2016 |
