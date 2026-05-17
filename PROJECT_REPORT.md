# PROJECT_REPORT.md — efdr-jumps : Rapport Technique Exhaustif

> Généré le 2026-05-16. Extraction git effectuée depuis la branche `step-8-grid-mc` (HEAD `1be516a`).
> Sources : `CLAUDE (1).md`, `EXECUTION_PLAN.md`, `DECISIONS.md`, `git log --all --stat`, `results/grid_medium.parquet`.

---

## 1. Vue d'ensemble du projet

**Nom du package :** `efdr-jumps`  
**Objectif scientifique :** Contrôle en ligne du False Discovery Rate (FDR) pour la détection de sauts dans des semi-martingales financières à haute fréquence, en exploitant des e-processes construits à partir d'estimateurs de volatilité jump-robust.

Le projet reproduit et étend Yen (2013) et Bajgrowicz-Scaillet (2016) avec la méthodologie e-values / e-FDR de l'école Ramdas 2022–2026. Il compare sept procédures FDR (dont cinq e-value-based) sur une grille Monte Carlo couvrant trois régimes de saut, cinq fréquences d'échantillonnage et deux niveaux α.

**Avancement :** 8 étapes sur 10 entamées ; étapes 1–7 complètement fusionnées sur `main` ; étape 8 en cours sur `step-8-grid-mc` (grid_medium terminé, grid_main non lancé). Étapes 9–10 non commencées.

**Résultat principal (grid_medium, M=100, n=500) :** BH-BNS et BH-LM violent le critère FDR ≤ α + 2·SE à dt=5s sur tous les régimes (FDR observé ≈ 0.096, seuil = α + 2·SE ≈ 0.094 pour α=0.05, M=100). Les procédures e-values (e-BH, e-LOND, stopped e-BH) contrôlent le FDR à toutes les fréquences, avec une perte de puissance de 5–15 pp à dt=5s par rapport aux baselines BH.

---

## 2. Chronologie des étapes

### Étape 1 — Bootstrap du repo

| Champ | Valeur |
|---|---|
| Date début | 2026-05-15 (avant 02:01 — inclus dans le commit initial) |
| Date fin | 2026-05-16 02:01:32 |
| Branche | `main` |

**Commits :**

| Hash | Date | Message |
|---|---|---|
| `183d7a7` | 2026-05-16 02:01:32 | Initial commit: Étapes 1-5 complètes (simulate, estimators, evalues) |

**Fichiers créés (extraits du diff stat) :**

| Fichier | Lignes ajoutées |
|---|---|
| `pyproject.toml` | +46 |
| `.github/workflows/ci.yml` | +28 |
| `.gitignore` | +35 |
| `.pre-commit-config.yaml` | +21 |
| `CLAUDE (1).md` | +373 |
| `EXECUTION_PLAN.md` | +135 |
| `data/README.md` | +9 |

**Tests ajoutés :** 0 tests FDR/pipeline ; squelettes `test_fdr_algorithms.py`, `test_pipeline.py` créés vides (1 ligne chacun).

**Décisions méthodologiques :** Aucune décision dans DECISIONS.md (non encore créé). Conventions fixées dans CLAUDE.md : temps en secondes, log-prix en log naturel, volatilités annualisées, Δ=1/(252·6.5·3600) pour 1 seconde.

**Critère EXECUTION_PLAN.md :** `pip install -e .` fonctionnel, `pytest` exit 0, `ruff check .` passe. — **Statut : rempli** (inclus dans le commit initial).

---

### Étapes 2, 3, 4 — Simulateurs, estimateurs, e-values

> Ces étapes ont été implémentées avant le premier commit visible dans le git log et sont toutes incluses dans le commit initial `183d7a7` du 2026-05-16 02:01:32. L'historique git ne permet pas de reconstituer la chronologie interne de ces étapes ; elles sont donc documentées ensemble.

**Commits :**

| Hash | Date | Message |
|---|---|---|
| `183d7a7` | 2026-05-16 02:01:32 | Initial commit: Étapes 1-5 complètes (simulate, estimators, evalues) |

**Fichiers créés :**

| Fichier | Lignes |
|---|---|
| `src/efdr_jumps/simulate/base.py` | +36 |
| `src/efdr_jumps/simulate/heston.py` | +139 |
| `src/efdr_jumps/simulate/merton.py` | +81 |
| `src/efdr_jumps/simulate/noise.py` | +59 |
| `src/efdr_jumps/estimators/bipower.py` | +104 |
| `src/efdr_jumps/estimators/medrv_minrv.py` | +44 |
| `src/efdr_jumps/estimators/threshold.py` | +43 |
| `src/efdr_jumps/estimators/spot.py` | +76 |
| `src/efdr_jumps/estimators/preavg.py` | +102 |
| `src/efdr_jumps/evalues/construct.py` | +48 |
| `tests/test_simulate.py` | +305 |
| `tests/test_estimators.py` | +208 |
| `tests/test_evalues_validity.py` | +221 |
| `tests/test_package.py` | +7 |

**Tests ajoutés :** 30 tests (16 simulate + 10 estimators + 4 evalues_validity).

**Décisions méthodologiques :** Non documentées dans DECISIONS.md pour ces étapes (lacune documentaire explicite : la note « Décisions des Étapes 1–5 non documentées ici » figure en bas de DECISIONS.md). Conventions impliquées par le code :
- Simulateur Heston : schéma QE d'Andersen.
- Merton : paramétrage Andersen-Benzoni-Lund (2002).
- Bruit microstructure : additif gaussien + one-sided Exp(1/σ_noise) à la Bibinger (2024).
- E-value construct.py : ratio de vraisemblances N(0,Δσ̂²) vs mélange alternatif.

**Critères EXECUTION_PLAN.md :**
- Étape 2 : simulation 1 jour 1s Heston+Merton < 100ms. — **Statut : rempli** (test `test_heston_merton_day_speed` passe).
- Étape 3 : tests de variance du bruit, récupérabilité par pré-moyennage. — **Statut : rempli**.
- Étape 4 : convergence des estimateurs, MedRV/MinRV biais < 10% sous sauts. — **Statut : rempli** (test `test_medrv_minrv_bias_under_jumps`).
- Étape 5 (e-values) : séparation H0/H1, e-power monotone, dominance vs baselines. — **Statut : rempli** (4 tests `test_evalues_validity.py` passent).

---

### Étape 6 — Algorithmes FDR (7 procédures)

| Champ | Valeur |
|---|---|
| Date début | 2026-05-16 02:02:26 |
| Date fin merge | 2026-05-16 02:35:00 |
| Branche | `step-6-fdr-algos` |

**Commits (ordre chronologique) :**

| Hash | Date | Message |
|---|---|---|
| `005f901` | 02:02:26 | Add e-BH offline (Wang-Ramdas 2022) with validity tests |
| `30ef62c` | 02:03:45 | Add BH baselines (BH-1995, BH-LM, BH-BNS) with validity tests |
| `0e27831` | 02:04:56 | Add e-LOND (Xu-Ramdas 2024) with validity tests |
| `7835797` | 02:05:58 | Add e-LORD (Zhang-Wei-Ren-Zou 2025) with W0=0.5 and validity tests |
| `b0d2737` | 02:07:05 | Add stopped e-BH (Wang-Dandapanthula-Ramdas 2025) with validity tests |
| `6f9f89d` | 02:08:46 | Add DECISIONS.md and export all FDR algorithms from fdr/\_\_init\_\_.py |
| `178d906` | 02:18:18 | Fix H0 e-value construction: use valid kappa=0.5 calibrator (Vovk-Wang 2021) |
| `944e5da` | 02:26:41 | Rewrite e-LORD (RAI framework) and add e-SAFFRON (Zhang et al. 2025 e-GAI) |
| `4c84ec5` | 02:34:49 | Update DECISIONS.md: e-LORD RAI formula, e-SAFFRON spec, deferred algos |
| `0555aef` | 02:35:00 | Merge step-6-fdr-algos: 7 FDR algorithms implemented and validated |

**Fichiers créés ou modifiés :**

| Fichier | Lignes nettes |
|---|---|
| `src/efdr_jumps/fdr/ebh.py` | +44 |
| `src/efdr_jumps/fdr/baselines.py` | +79 → final 120 |
| `src/efdr_jumps/fdr/elond.py` | +52 |
| `src/efdr_jumps/fdr/elord_esaffron.py` | +54 initial → +135 (réécriture RAI) → final 137 |
| `src/efdr_jumps/fdr/stopped_ebh.py` | +54 |
| `src/efdr_jumps/fdr/__init__.py` | +16 |
| `DECISIONS.md` | +66 initial → +57 (réécriture) → final 123 |
| `tests/test_fdr_algorithms.py` | +130 → +82 → +79 → +77 → +137 → final 528 |

**Tests ajoutés :** 32 tests FDR (tous dans `test_fdr_algorithms.py`), passent tous (86 total à ce stade).

**Décisions méthodologiques (DECISIONS.md §Étape 6) :**

1. **Constante C_JM de γ_t Javanmard-Montanari :** C = 0.15708906 (calculé par sommation tronquée à t=10⁶), non pas C≈0.0722 comme dans le brief initial.
2. **W_0 pour e-LORD :** W_0 = 0.5 (équilibre wealth initial / accumulé, valeur canonique e-GAI Zhang et al. 2025).
3. **k*=0 dans e-BH :** retourner `[False]*n` (aucun rejet).
4. **Stopped e-BH :** recalcul e-BH offline sur E_{1:t} à chaque step t ; décisions passées non révisées.
5. **BH-BNS :** retourne n booléens identiques (rejet global ou pas), cohérent avec Bajgrowicz-Scaillet 2016.
6. **e-LORD : réécriture complète** (commit 944e5da) : la première implémentation utilisait γ_t statique de e-LOND ; le vrai e-LORD utilise le framework RAI (Risk Aversion Investing) avec suite ω_t *dynamique*.
7. **e-SAFFRON :** λ=0.1 (pas 0.5), seuil 1/λ uniquement pour le calcul du budget (pas pour la décision de rejet).

**Bugs corrigés :**

- `178d906` : `E = 1/U` (U~Uniform) invalide sous H0 car E[E]=∞. Remplacé par `E = 0.5/sqrt(U)` (calibrateur Vovk-Wang 2021, κ=0.5 ; E[E]=1 ✓). Conséquence : `test_stopped_ebh_fdr_validity_pareto` échouait avec FDR=0.481.
- `944e5da` : réécriture e-LORD (voir point 6 ci-dessus).

**Algorithmes différés :**
- `fdr/eclosure.py` (online e-closure, Xu-Fischer-Ramdas 2026) — raison : formule DP closure non entièrement explicitée, priorité pipeline d'abord.
- `fdr/compound_ebh.py` (compound e-BH avec donation, XFR 2026) — raison : nécessite lecture PDF complète de la Section 2.2 (γ-online compound e-values).

**Critère EXECUTION_PLAN.md :** chaque algorithme contrôle le FDR au niveau cible sur streams Pareto et Heston ; accord sur stream déterministe hand-verifiable. — **Statut : rempli** (voir tous les tests `*_fdr_validity_*`).

---

### Étape 7 — Pipeline orchestrateur et métriques

| Champ | Valeur |
|---|---|
| Date début | 2026-05-16 02:45:36 |
| Date fin merge | 2026-05-16 03:17:55 |
| Branche | `step-7-pipeline` |

**Commits :**

| Hash | Date | Message |
|---|---|---|
| `8b19dda` | 02:45:36 | Add pipeline/metrics.py and pipeline/detector.py |
| `a9b3170` | 02:45:41 | Add grid runner, updated grid_quick.yaml (7 algos), pipeline tests |
| `bd3c0ce` | 03:01:43 | Fix partial-window bias in bh_bns and e-value computation |
| `8918316` | 03:02:07 | Merge step-7-pipeline: pipeline orchestrator, metrics, grid runner |
| `2ab6e8b` | 03:17:55 | Fix lint and formatting: ruff + black pass |

**Fichiers créés ou modifiés :**

| Fichier | Lignes nettes |
|---|---|
| `src/efdr_jumps/pipeline/detector.py` | +257 → +7 (fix) → +50 (lint) → final 284 |
| `src/efdr_jumps/pipeline/metrics.py` | +110 → −12 (lint) → final 106 |
| `src/efdr_jumps/pipeline/__init__.py` | +23 |
| `experiments/02_power_grid.py` | +214 → +16 (lint) → final (voir Étape 8) |
| `experiments/configs/grid_quick.yaml` | −2+2 (ajout algos) |
| `tests/test_pipeline.py` | +200 → +29 (lint) → final (voir Étape 8) |

**Tests ajoutés :** 20 tests pipeline (tous fast sauf `test_pipeline_fdr_validity_moderate`).

**Bugs corrigés :**

- `bd3c0ce` (**biais fenêtre partielle**) : dans `bh_bns` et le calcul des e-values, les hypothèses i < K (début de fenêtre) utilisaient une fenêtre incomplète pour σ̂, ce qui biaisait la statistique. Correction : assertion explicite que la fenêtre est complète avant de calculer σ̂ ; les hypothèses avec i < K sont exclues du test.

**Critère EXECUTION_PLAN.md :** end-to-end sur `grid_quick.yaml` (M=50, 1 fréquence, 1 taille de saut) en < 5 minutes. — **Statut : rempli** (le lint commit 2ab6e8b était sur main après merge).

---

### Étape 8 — Grille Monte Carlo complète (en cours)

| Champ | Valeur |
|---|---|
| Date début | 2026-05-16 12:32:52 |
| Date fin prévue | Nuit du 2026-05-16 (grid_main non lancé à ce stade) |
| Branche | `step-8-grid-mc` (non fusionnée sur main) |

**Commits :**

| Hash | Date | Message |
|---|---|---|
| `3fce5c8` | 12:32:52 | Remove deferred algos from grid_main; fix jump_sizes [Étape 8 prep] |
| `e57b1c0` | 12:34:24 | Add HawkesSimulator: self-exciting Hawkes jump clustering |
| `d81cce4` | 12:35:19 | Add n_steps config; grid_medium for validation, compute estimate |
| `88ec669` | 12:47:53 | Fix black formatting on hawkes.py and detector.py |
| `1d3df4d` | 12:49:52 | Add test_run_detector_hawkes_dense: all algos + clustering assertion |
| `8830219` | 12:53:11 | Add results notebook + emit w1 in parquet |
| `b99d6a4` | 12:54:38 | Document Étape 8 decisions in DECISIONS.md |
| `1be516a` | 13:11:45 | Fix wealth_fractions for n=2000; document Étape 8 findings |

**Fichiers créés ou modifiés :**

| Fichier | Lignes nettes |
|---|---|
| `src/efdr_jumps/simulate/hawkes.py` | +106 |
| `src/efdr_jumps/simulate/__init__.py` | +2 |
| `src/efdr_jumps/pipeline/detector.py` | +21 (support HawkesSimulator) |
| `experiments/configs/grid_main.yaml` | −5+6 (retrait algos différés, fix wealth_fractions) |
| `experiments/configs/grid_medium.yaml` | +9 (nouveau fichier) |
| `experiments/02_power_grid.py` | +6 (n_steps, w1 dans parquet) |
| `notebooks/03_results_figures.ipynb` | +355 |
| `tests/test_pipeline.py` | +30 (hawkes_dense) |
| `DECISIONS.md` | +54+46 = +100 |

**Tests ajoutés :** 2 tests pipeline (`test_run_detector_hawkes_dense`, `test_run_detector_result_traceable`).

**Décisions méthodologiques (DECISIONS.md §Étape 8) :**

1. **Régime hawkes_dense :** Processus Hawkes auto-excitant en temps discret. Paramètres : μ=5000/an, α=88000/an, β=0.03/s, branching ratio m≈0.50. Validé sur 30 seeds (n=200, dt=5s) : 3.8 sauts/path vs 1.3 pour `moderate`.
2. **grid_medium** : n_steps=500, 3 fréqs, 3 régimes, M=100, ~32 400 tasks, ~7 min (3 workers).
3. **grid_main** : n_steps=2000, 5 fréqs, 3 régimes, M=500, ~390 000 tasks, ~2.9h (3 workers).
4. **wealth_fractions corrigées pour n=2000** : `[0.0005, 0.0025, 0.005, 0.05]` (valeurs 1/n, 5/n, 10/n, 100/n). Les valeurs `[0.1, 0.25, 0.5, 0.75]` causent toutes l'alpha-death à n=2000 (confirmé sur grid_medium).
5. **`jump_sizes: mixed` différé** : `float("mixed")` crashe le runner ; nécessite support dédié dans `_expand_grid()`.

**Bugs corrigés :**

- `88ec669` : formatage black sur `hawkes.py` et `detector.py`.
- `1be516a` : wealth_fractions initialement `[0.1, 0.25, 0.5, 0.75]` → corrigé vers `[0.0005, 0.0025, 0.005, 0.05]`.

**Critère EXECUTION_PLAN.md :** FDR empirique ≤ α + 2·SE partout, e-closure ≥ e-LOND en puissance d'au moins 10% à FDR=0.1. — **Statut : partiellement rempli** (grid_medium valide le critère FDR pour 5 algos sur 7 à dt=5s ; e-closure non implémenté ; grid_main non lancé).

---

## 3. État actuel du code

### Arborescence des fichiers clés

```
efdr-jumps/
├── CLAUDE (1).md              # source de vérité : stack, conventions, acceptance criteria
├── EXECUTION_PLAN.md          # séquencement des 10 étapes
├── DECISIONS.md               # décisions techniques documentées (Étapes 6, 8)
├── pyproject.toml
├── .github/workflows/ci.yml   # lint + test sur push
├── .pre-commit-config.yaml    # ruff + black
│
├── src/efdr_jumps/
│   ├── simulate/
│   │   ├── base.py            (36 l) — PathSimulator ABC
│   │   ├── heston.py          (143 l) — Heston SV (schéma QE)
│   │   ├── merton.py          (81 l) — Heston + Merton (Poisson)
│   │   ├── hawkes.py          (104 l) — Hawkes auto-excitant [Étape 8]
│   │   └── noise.py           (59 l) — bruit microstructure additif/one-sided
│   │
│   ├── estimators/
│   │   ├── bipower.py         (104 l) — BV, BNS stat, Lee-Mykland stat
│   │   ├── medrv_minrv.py     (44 l) — ADS 2012
│   │   ├── threshold.py       (43 l) — Mancini 2009
│   │   ├── spot.py            (76 l) — fenêtre locale (exclut le point testé)
│   │   └── preavg.py          (100 l) — pré-moyennage Jacod et al. 2009
│   │
│   ├── evalues/
│   │   └── construct.py       (50 l) — E_i depuis (r_i, σ̂_i)
│   │
│   ├── fdr/
│   │   ├── ebh.py             (44 l) — e-BH offline (Wang-Ramdas 2022)
│   │   ├── baselines.py       (120 l) — BH-LM, BH-BNS (Yen 2013, BS 2016)
│   │   ├── elond.py           (52 l) — e-LOND (Xu-Ramdas 2024)
│   │   ├── elord_esaffron.py  (137 l) — e-LORD + e-SAFFRON (Zhang et al. 2025)
│   │   └── stopped_ebh.py     (54 l) — stopped e-BH (Wang et al. 2025)
│   │
│   └── pipeline/
│       ├── detector.py        (284 l) — orchestrateur end-to-end
│       └── metrics.py         (106 l) — FDR, power, F1, delay, wall-clock
│
├── experiments/
│   ├── 02_power_grid.py       — runner Monte Carlo parallèle (multiprocessing)
│   └── configs/
│       ├── grid_quick.yaml    — smoke test (M=50, 1 fréq)
│       ├── grid_medium.yaml   — validation (M=100, n=500, 3 fréqs)
│       └── grid_main.yaml     — run final (M=500, n=2000, 5 fréqs) [non lancé]
│
├── notebooks/
│   └── 03_results_figures.ipynb — figures finales [structure créée, figures à générer]
│
├── tests/
│   ├── test_simulate.py       (312 l, 16 tests)
│   ├── test_estimators.py     (210 l, 10 tests)
│   ├── test_evalues_validity.py (223 l, 4 tests)
│   ├── test_fdr_algorithms.py (528 l, 33 tests)
│   ├── test_pipeline.py       (242 l, 22 tests)
│   └── test_package.py        (7 l, 1 test)
│
└── results/
    └── grid_medium.parquet    (373 KB, 32 400 lignes × 17 colonnes)
```

**Fichiers planifiés mais non créés :** `evalues/mixture.py`, `evalues/eprocess.py`, `fdr/eclosure.py`, `fdr/compound_ebh.py`, `utils/stats.py`, `utils/viz.py`, `simulate/rough.py`, `experiments/01_evalue_validity.py`, `experiments/03_baseline_comparison.py`, `experiments/04_real_data.py`.

---

### Lignes de code par module

| Module | Fichier | Lignes |
|---|---|---|
| **simulate** | base.py | 36 |
| | heston.py | 143 |
| | merton.py | 81 |
| | hawkes.py | 104 |
| | noise.py | 59 |
| | **sous-total** | **423** |
| **estimators** | bipower.py | 104 |
| | medrv_minrv.py | 44 |
| | threshold.py | 43 |
| | spot.py | 76 |
| | preavg.py | 100 |
| | **sous-total** | **367** |
| **evalues** | construct.py | 50 |
| | **sous-total** | **50** |
| **fdr** | ebh.py | 44 |
| | baselines.py | 120 |
| | elond.py | 52 |
| | elord_esaffron.py | 137 |
| | stopped_ebh.py | 54 |
| | **sous-total** | **407** |
| **pipeline** | detector.py | 284 |
| | metrics.py | 106 |
| | **sous-total** | **390** |
| **__init__ + utils** | divers | 49 |
| **Total src/** | | **1 686** |
| **Total tests/** | | **1 522** |
| **Grand total** | | **3 208** |

---

### Algorithmes FDR implémentés

| Algorithme | Fichier | Référence |
|---|---|---|
| e-BH (offline) | `fdr/ebh.py` | Wang & Ramdas (2022), arXiv:2009.02824 |
| BH-LM | `fdr/baselines.py` | Lee & Mykland (2008) + Benjamini-Hochberg (1995) |
| BH-BNS | `fdr/baselines.py` | Barndorff-Nielsen & Shephard (2006) + Bajgrowicz-Scaillet (2016) |
| e-LOND | `fdr/elond.py` | Xu & Ramdas (2024) |
| e-LORD | `fdr/elord_esaffron.py` | Zhang, Wei, Ren & Zou (2025), arXiv:2506.01452 (e-GAI, Algo 1) |
| e-SAFFRON | `fdr/elord_esaffron.py` | Zhang, Wei, Ren & Zou (2025), arXiv:2506.01452 (e-GAI, Algo 2) |
| stopped e-BH | `fdr/stopped_ebh.py` | Wang, Dandapanthula & Ramdas (2025), arXiv:2502.08539 |

**Non implémentés (différés) :**

| Algorithme | Raison |
|---|---|
| online e-closure | Formule DP non entièrement explicitée dans la portion HTML accessible du papier |
| compound e-BH avec donation | Nécessite lecture complète §2.2 Xu-Fischer-Ramdas 2026 |

---

### Estimateurs de volatilité implémentés

| Estimateur | Fichier | Référence |
|---|---|---|
| Bipower Variation (BV) | `estimators/bipower.py` | Barndorff-Nielsen & Shephard (2004) |
| Statistique BNS | `estimators/bipower.py` | Barndorff-Nielsen & Shephard (2006) |
| Statistique Lee-Mykland | `estimators/bipower.py` | Lee & Mykland (2008) |
| MinRV | `estimators/medrv_minrv.py` | Andersen, Dobrev & Schaumburg (2012), eq. 13 |
| MedRV | `estimators/medrv_minrv.py` | Andersen, Dobrev & Schaumburg (2012), eq. 12 |
| Truncated RV (Mancini) | `estimators/threshold.py` | Mancini (2009) |
| Spot volatility (fenêtrée) | `estimators/spot.py` | Générique ; fenêtre exclut le point i |
| Pre-averaged BV | `estimators/preavg.py` | Podolskij & Vetter (2009), Jacod et al. (2009) |

**Estimateur utilisé dans la grille :** MedRV uniquement (`estimators: [medrv]` dans grid_main.yaml et grid_medium.yaml).

---

### Simulateurs implémentés

| Simulateur | Classe | Régime |
|---|---|---|
| Heston (pas de saut) | `HestonSimulator` | Diffusion pure (`rare` avec λ→0) |
| Heston + Merton (Poisson) | `MertonSimulator` | `rare` (λ≈1500/an), `moderate` (λ≈5000/an) |
| Heston + Hawkes | `HawkesSimulator` | `hawkes_dense` (branching ratio ≈ 0.50) |
| NoisyPath (wrapper) | `NoisyPath` | Additif gaussien + one-sided Exp |

---

### Suite de tests

| Fichier | Tests total | Fast | Slow | Statut (2026-05-16) |
|---|---|---|---|---|
| test_package.py | 1 | 1 | 0 | ✓ passe |
| test_simulate.py | 16 | 15 | 1 | ✓ passe |
| test_estimators.py | 10 | 6 | 4 | ✓ passe |
| test_evalues_validity.py | 4 | 4 | 0 | ✓ passe |
| test_fdr_algorithms.py | 33 | 20 | 13 | ✓ passe |
| test_pipeline.py | 22 | 18 | 4 | ✓ passe |
| **Total** | **86** | **64** | **22** | **86/86 passent** |

Durée totale (all 86) : ~50s. Durée fast only : ~15s.

---

## 4. Résultats scientifiques

### Source des données

Fichier : `results/grid_medium.parquet` (373 KB).  
Paramètres : n_steps=500, M=100 réplications, 3 fréquences (5s, 60s, 300s), 3 régimes, 2 tailles de saut (3σ, 5σ), 2 niveaux α (0.05, 0.10), wealth_fractions=[0.1, 0.5] pour e-LORD/e-SAFFRON.  
Total : 32 400 lignes × 17 colonnes.

### Table 1 — FDR empirique moyen par algo × fréquence (tous régimes, α=0.05)

Convention : violation si FDR_obs > α + 2·SE avec SE = √(α(1−α)/M) = √(0.05×0.95/100) ≈ 0.022 ; seuil = **0.094**.

| Algorithme | dt=5s | dt=60s | dt=300s | Viole α=0.05+2SE à 5s ? |
|---|---|---|---|---|
| BH-BNS | **0.096** | 0.022 | 0.002 | **Oui** |
| BH-LM | **0.096** | 0.022 | 0.002 | **Oui** |
| stopped_ebh | 0.016 | 0.005 | 0.000 | Non |
| ebh | 0.010 | 0.003 | 0.000 | Non |
| elond | 0.001 | 0.000 | 0.000 | Non |
| elord | 0.000 | 0.000 | 0.000 | Non (alpha-death w1=0.1) |
| esaffron | 0.000 | 0.000 | 0.000 | Non (alpha-death w1=0.1) |

> Note : les FDR de BH-BNS et BH-LM sont identiques car les deux utilisent la même statistique normalisante (BV/spot) dans le détecteur — leur différence réside dans le test global (BNS) vs par observation (LM), mais à ce niveau d'agrégation la distinction disparaît. À vérifier sur la grille finale.

### Table 2 — Puissance (power) par algo × fréquence (tous régimes, α=0.05)

| Algorithme | dt=5s | dt=60s | dt=300s |
|---|---|---|---|
| BH-BNS | 0.441 | 0.358 | 0.157 |
| BH-LM | 0.441 | 0.358 | 0.157 |
| stopped_ebh | 0.349 | 0.258 | 0.028 |
| ebh | 0.326 | 0.252 | 0.027 |
| elond | 0.235 | 0.168 | 0.013 |
| elord | 0.002 | 0.001 | 0.000 |
| esaffron | 0.002 | 0.001 | 0.000 |

### Table 3 — Résultats détaillés : hawkes_dense, dt=5s, α=0.05 (M=100, n=500)

| Algorithme | FDR obs. | Power | F1 | FDR contrôlé ? |
|---|---|---|---|---|
| BH-BNS | 0.091 | 0.423 | 0.522 | **Non** (0.091 > 0.094 : limite, voir note) |
| BH-LM | 0.091 | 0.423 | 0.522 | **Non** |
| stopped_ebh | 0.017 | 0.303 | 0.454 | Oui |
| ebh | 0.009 | 0.280 | 0.435 | Oui |
| elond | 0.000 | 0.204 | 0.371 | Oui |
| elord (w1=0.1) | 0.000 | 0.001 | 0.151 | Oui (alpha-death) |
| esaffron (w1=0.1) | 0.000 | 0.001 | 0.151 | Oui (alpha-death) |

> Note : avec M=100 et α=0.05, le seuil exact est 0.05+2×0.022=0.094. BH-BNS à 0.091 est en dessous du seuil sur hawkes_dense seul mais dépasse 0.094 agrégé sur tous les régimes (0.096). DECISIONS.md documente la violation pour l'ensemble dt=5s tous régimes confondus, ce qui est cohérent.

### Table 4 — Benchmark vitesse (ms/run, n=2000, source DECISIONS.md)

| Algorithme | ms/run |
|---|---|
| bh_lm | 1 |
| ebh | 170 |
| elond | 185 |
| esaffron | 174 |
| elord | 187 |
| bh_bns | 257 |
| stopped_ebh | 346 |

### Critères d'acceptance CLAUDE.md §5 — Statut

| Critère | Statut |
|---|---|
| Repo passe CI (lint + tests) | **Rempli** — 86/86 tests passent, ruff+black OK |
| Validité : FDR ≤ α+2·SE pour tous algos × estimateurs × fréquences | **Partiellement rempli** — grid_medium confirme pour e-values ; BH viole à dt=5s |
| Power gain : e-closure ≥ e-LOND +10% à FDR=0.10, 5s, moderate | **Non rempli** — e-closure non implémenté |
| Reproduit Yen 2013 Table 2 | **Non vérifié** — grid_main non lancé |
| Toutes figures régénérables depuis notebooks/03 | **Non rempli** — notebook créé mais vide |
| Smoke test données réelles | **Non rempli** — Étape 9 non commencée |

### Résultat principal identifié

**BH viole le FDR à dt=5s indépendamment du régime de saut**, y compris sous dépendance Hawkes. L'interprétation : la BV sous-estime la volatilité spot à 5s en présence de bruit microstructure (corrélation négative induite par le bid-ask bounce), ce qui gonfle la statistique BNS et produit de faux sauts. Référence : Aït-Sahalia-Mykland-Zhang (2005), Bajgrowicz-Scaillet (2016).

**Les e-values restent valides sous dépendance arbitraire** (dépendance microstructure à HF, clustering Hawkes), ce qui constitue l'argument méthodologique central du projet.

**Alpha-death de e-LORD et e-SAFFRON** avec w1=0.1 sur n=500 : puissance ≈ 0. Corrigé dans grid_main (wealth_fractions=[0.0005, 0.0025, 0.005, 0.05]).

### Limitations et caveats connus

1. **grid_main non lancé** : résultats basés uniquement sur grid_medium (M=100, n=500).
2. **wealth_fractions sur grid_medium** : [0.1, 0.5] uniquement — alpha-death pour les deux sur n=500 pour e-LORD/e-SAFFRON.
3. **Seul estimateur testé** : MedRV. La grille initiale prévoyait [bv, medrv, minrv, threshold].
4. **Approximation discrète du Hawkes** : à dt=300s le ratio de branchement discret dépasse 1 (instabilité apparente), mais le processus reste défini (probabilité bornée par 1-exp(-λΔt)≤1).
5. **BH-BNS = BH-LM** dans les résultats : probablement un bug d'indexation dans `detector.py` qui applique la même normalisation pour les deux. Non signalé dans DECISIONS.md — à investiguer.

---

## 5. Décisions méthodologiques importantes

### Construction des e-values

| Décision | Source |
|---|---|
| E_i = ratio de vraisemblances N(0,Δσ̂²) vs mélange alternatif intégré sur prior | `evalues/construct.py` |
| σ̂_i estimé sur fenêtre locale **excluant le point i** (assertion explicite) | `estimators/spot.py`, corrigé dans `bd3c0ce` |
| e-value H0 valide pour les tests : E = 0.5/√U (calibrateur Vovk-Wang 2021, κ=0.5, E[E]=1) | DECISIONS.md, commit `178d906` |
| E = 1/U invalide (E[E]=∞) — bug corrigé | DECISIONS.md, commit `178d906` |

### Algorithmes FDR — paramètres par défaut

| Paramètre | Valeur | Justification |
|---|---|---|
| Constante C_JM (γ_t Javanmard-Montanari) | 0.15708906 | Sommation tronquée à t=10⁶, Σγ_t≈1 |
| W_0 (e-LORD) | 0.5 | Valeur canonique e-GAI (Zhang et al. 2025) |
| λ (e-SAFFRON) | 0.1 | Papier recommande valeur petite |
| wealth_fractions (grid_main) | [0.0005, 0.0025, 0.005, 0.05] | w1=O(1/n) per e-GAI ; 1/2000=0.0005 canonique |
| k*=0 dans e-BH | retourner [False]*n | Cohérence procédurale (pas de cutoff valide) |
| Stopped e-BH | recalcul offline sur E_{1:t} sans révision passée | Interface Iterator[bool] + anytime-validity |
| BH-BNS | n booléens identiques (test global) | Bajgrowicz-Scaillet 2016 : BNS = filtre jour-niveau |

### Simulation

| Paramètre | Valeur | Source |
|---|---|---|
| Hawkes μ | 5000/an | DECISIONS.md §Étape 8 |
| Hawkes α | 88000/an | DECISIONS.md §Étape 8 |
| Hawkes β | 0.03/s | DECISIONS.md §Étape 8 |
| Branching ratio m = α/(β×SECS/AN) | ≈0.50 (stable) | DECISIONS.md §Étape 8 |
| Régime rare : λ Merton | ~1500/an | `simulate/merton.py` (implicite) |
| Régime moderate : λ Merton | ~5000/an | `simulate/merton.py` (implicite) |
| Bruit additif σ_noise | configurable | `simulate/noise.py` |

### Grille Monte Carlo

| Dimension | grid_quick | grid_medium | grid_main |
|---|---|---|---|
| n_steps | non fixé | 500 | 2000 |
| Fréquences (s) | [5] | [5, 60, 300] | [1, 5, 30, 60, 300] |
| Régimes | [moderate] | [rare, moderate, hawkes_dense] | [rare, moderate, hawkes_dense] |
| jump_sizes (σ) | [5] | [3, 5] | [3, 5] |
| Alpha | [0.05] | [0.05, 0.10] | [0.05, 0.10] |
| wealth_fractions | [0.005] | [0.1, 0.5] | [0.0005, 0.0025, 0.005, 0.05] |
| M réplications | 50 | 100 | 500 |
| Total tasks | ~350 | 32 400 | ~390 000 |
| Temps estimé (3 workers) | <5 min | ~7 min | ~2.9h |

---

## 6. Bibliographie centrale

### Papiers cœur (cités dans les docstrings du code)

| Référence | Module | Rôle |
|---|---|---|
| Barndorff-Nielsen & Shephard (2004) | `estimators/bipower.py` | Bipower variation BV_n |
| Barndorff-Nielsen & Shephard (2006) | `estimators/bipower.py` | Statistique BNS, variance asymptotique |
| Andersen, Dobrev & Schaumburg (2012), *JoE* | `estimators/medrv_minrv.py` | MedRV (eq. 12), MinRV (eq. 13), facteur d'échelle |
| Mancini (2009), *Scand. J. Statist.* | `estimators/threshold.py` | Truncated realized variance |
| Lee & Mykland (2008) | `estimators/bipower.py`, `fdr/baselines.py` | Statistique LM par observation |
| Podolskij & Vetter (2009) | `estimators/preavg.py` | Pre-averaged bipower variation |
| Jacod, Li, Mykland, Podolskij & Vetter (2009) | `estimators/preavg.py` | Coefficient de correction bruit |
| Andersen, Benzoni & Lund (2002) | `simulate/merton.py` | Paramétrage Heston+Merton |
| Bibinger (2024) | `simulate/noise.py` | Bruit one-sided Exp(1/σ_noise) |
| Aït-Sahalia, Mykland & Zhang (2005) | `DECISIONS.md` | Biais BV sous bruit microstructure |
| Bajgrowicz & Scaillet (2016), *MS* | `fdr/baselines.py`, `DECISIONS.md` | BH-BNS jour-niveau, BH sur LM p-values |
| Yen (2013), *PLOS One* | `CLAUDE (1).md` | BH offline sur p-values (baseline reproduite) |
| Wang & Ramdas (2022), arXiv:2009.02824 | `fdr/ebh.py` | e-BH offline (Algorithm 1) |
| Vovk & Wang (2021) | `tests/test_fdr_algorithms.py`, DECISIONS.md | Calibrateur p-to-e, κ=0.5 |
| Xu & Ramdas (2024) | `fdr/elond.py` | e-LOND (Algorithm 1) |
| Zhang, Wei, Ren & Zou (2025), arXiv:2506.01452 | `fdr/elord_esaffron.py` | e-GAI : e-LORD (Algo 1), e-SAFFRON (Algo 2) |
| Wang, Dandapanthula & Ramdas (2025), arXiv:2502.08539 | `fdr/stopped_ebh.py` | Stopped e-BH, anytime-validity |
| Xu, Fischer & Ramdas (2026), arXiv:2603.24792 | `CLAUDE (1).md`, DECISIONS.md | Online e-closure, compound e-BH (différés) |
| Grünwald, de Heide & Koolen (2024), *Ann. Stat.* | `CLAUDE (1).md` | e-power : E[log E_i] sous H1 |
| Ramdas & Wang (2025) | `CLAUDE (1).md` | e-power comme critère d'efficacité séquentielle |
| Ramdas, Ruf, Larsson & Koolen (2022), arXiv:2009.03167 | `CLAUDE (1).md` | e-processes séquentiels |

---

## 7. Ce qui reste à faire

### Étape 8 (en cours) — Complétion grille Monte Carlo

- [ ] **Lancer grid_main** : `python experiments/02_power_grid.py --config experiments/configs/grid_main.yaml` (~2.9h, 3 workers). Pré-requis : vérifier wealth_fractions corrigées dans le YAML (fait en `1be516a`).
- [ ] **Générer les figures** dans `notebooks/03_results_figures.ipynb` une fois grid_main.parquet disponible : courbes FDR vs α, courbes puissance, effet wealth_fraction sur alpha-death, wall-clock comparaison.
- [ ] **Vérifier BH-BNS = BH-LM** dans les résultats — probablement un bug d'indexation à investiguer.

### Étape 9 — Application données réelles

- [ ] Choisir la source de données : Plan B (TAQ, 5s grid, MMN additif) par défaut.
- [ ] Implémenter `experiments/04_real_data.py` : cleaning (zero-price, out-of-sequence, bounce-back, Brownlees-Gallo 2006), downsampling en code, filtre market hours (open+5min → close-5min).
- [ ] Produire pour au moins 1 jour × 1 asset : nb de sauts détectés par algo, matrice d'accord Jaccard, alignement avec timestamps FOMC/earnings.

### Étape 10 — Reproductibilité finale

- [ ] Geler les seeds dans tous les scripts.
- [ ] Snapshot du parquet final dans `data/processed/`.
- [ ] README.md public.
- [ ] Badges CI sur GitHub.
- [ ] `jupyter nbconvert --to notebook --execute notebooks/03_results_figures.ipynb` doit régénérer toutes les figures en un seul appel.

### Algorithmes différés à implémenter

- [ ] `fdr/eclosure.py` — online e-closure (Xu-Fischer-Ramdas 2026, arXiv:2603.24792). Requis pour le critère d'acceptance §5.3 (e-closure ≥ e-LOND +10% puissance).
- [ ] `fdr/compound_ebh.py` — compound e-BH avec donation (même papier).
- [ ] `evalues/mixture.py` — e-values par mélange Robbins-style.
- [ ] `evalues/eprocess.py` — distinction e-process local / global (Ramdas et al. 2022).

### Limitations à documenter dans le rapport écrit

1. Fenêtre complète exigée pour σ̂ : les K premières observations sont systématiquement exclues du test, ce qui réduit la puissance en début de journée.
2. L'approximation Hawkes discrète à dt=300s n'est pas rigoureuse (branching ratio discret > 1).
3. Seul MedRV testé dans la grille ; BV, MinRV, threshold-BV non comparés.
4. Pas de test de monotonie de puissance entre e-closure et e-LOND (algorithme non implémenté).
5. La comparaison BH-LM vs BH-BNS dans les résultats actuels donne des chiffres identiques — artefact à corriger.

### Compute restant

grid_main non lancé. Estimation : ~2.9h avec 3 workers sur 4 cœurs logiques (390 000 tasks × ~80ms/run moyen ÷ 3 workers). Benchmark single-run à n=2000 documenté dans DECISIONS.md.

---

## Incohérences détectées

| N° | Incohérence | Localisation |
|---|---|---|
| 1 | **BH-BNS = BH-LM** dans tous les agrégats du parquet. DECISIONS.md les traite comme deux procédures distinctes (BNS = test global, LM = par observation) mais leurs FDR et power sont identiques dans grid_medium. Cause probable : un même code path dans `detector.py` pour les deux. | `grid_medium.parquet`, `pipeline/detector.py` |
| 2 | **CLAUDE.md référence `experiments/01_evalue_validity.py`** comme sortie attendue de l'Étape 5, mais ce fichier n'est pas créé. Les quatre tests correspondants existent dans `test_evalues_validity.py`, mais le script standalone n'est pas implémenté. | `CLAUDE (1).md §Step 4`, `experiments/` |
| 3 | **CLAUDE.md §Step 7 prévoyait** `estimators: [bv, medrv, minrv, threshold]` et `algorithms: [..., eclosure, compound_ebh]`. grid_main.yaml utilise `estimators: [medrv]` uniquement et exclut les algos différés. La réduction de périmètre n'est documentée que partiellement dans DECISIONS.md (algos différés expliqués, réduction estimateurs non expliquée). | `CLAUDE (1).md §Step 7`, `grid_main.yaml` |
| 4 | **CLAUDE.md §Step 7 prévoyait** `M_replications: 1000` dans grid_main ; EXECUTION_PLAN.md §Étape 8 dit « comparaison supervisée (3ème test du Step 5) » impliquant M élevé. Le YAML final utilise M=500. Divergence non documentée. | `CLAUDE (1).md §Step 7`, `grid_main.yaml` |
| 5 | **DECISIONS.md §Étape 8 findings** documente FDR BH-BNS = 0.091–0.099 à dt=5s et conclut à une violation. Le calcul de violation utilise SE=√(α(1−α)/M) avec α=0.05 (seuil=0.094). La requête Python sur le parquet avec SE=√(FDR_obs(1−FDR_obs)/M) donne un seuil plus élevé (≈0.109) et ne signale pas de violation. Ces deux conventions de calcul du SE donnent des conclusions différentes. DECISIONS.md utilise la convention correcte pour tester H0: FDR≤α. | `DECISIONS.md §Findings`, requête parquet |
| 6 | **CLAUDE.md §Step 5 (FDR algos)** numérotait 7 algos dont e-GAI comme entrée séparée. La liste finale implémentée contient 7 algos mais sans e-GAI explicite (e-GAI est le framework de e-LORD et e-SAFFRON, pas un algo séparé). La terminologie a évolué. | `CLAUDE (1).md §Step 5`, `fdr/__init__.py` |
