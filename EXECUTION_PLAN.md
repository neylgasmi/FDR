# EXECUTION_PLAN.md — 10 étapes pour compléter `efdr-jumps`

> Plan d'exécution opérationnel. Chaque étape correspond à une (ou plusieurs) session Claude Code et produit un PR mergeable. Référez-vous à `CLAUDE.md` pour le contenu détaillé de chaque module — ce document ne fait qu'ordonner *quand* faire *quoi*.

---

## Étape 1 — Bootstrap du repo

**Quand :** jour 1, première session.
**Quoi :** Step 1 du CLAUDE.md.
**Sortie attendue :** repo installable, CI verte, pre-commit configuré, structure de dossiers vide en place.
**Critère pour passer à l'étape suivante :** `pip install -e .` fonctionne, `pytest` exit 0 sur zéro test, `ruff check .` passe, push initial sur `main` accepté par les hooks GitHub Actions.
**Durée estimée :** 1 session de 30-45 min.

---

## Étape 2 — Simulateurs Heston puis Heston+Merton

**Quand :** une fois le bootstrap mergé.
**Quoi :** Step 2 du CLAUDE.md, sous-steps 1 → 2 → 3 (`base.py`, `heston.py`, `merton.py`).
**Sortie attendue :** trois modules dans `simulate/`, leurs tests KS et de récupération exacte des jump indices, un notebook de sanity check (`00_simulation_sanity_checks.ipynb`) qui affiche un sample path pour chaque DGP.
**Critère pour passer à l'étape suivante :** simulation d'un jour 1-seconde Heston+Merton en < 100ms, tests verts, figures du notebook visuellement plausibles (clustering vol, sauts visibles, distribution gaussienne des incréments hors saut).
**Durée estimée :** 1 à 2 sessions.

---

## Étape 3 — Bruit microstructure (et rough en option différée)

**Quand :** après Étape 2 validée.
**Quoi :** Step 2 sous-steps 4 (`noise.py`). Ne pas faire `rough.py` maintenant — c'est explicitement optionnel et différé.
**Sortie attendue :** wrapper `NoisyPath` qui prend un `PathSimulator` et y ajoute soit du bruit additif gaussien soit du bruit one-sided à la Bibinger 2024. Tests : le bruit a la variance théorique attendue, le signal reste récupérable par pré-moyennage.
**Critère pour passer à l'étape suivante :** tests verts, possibilité de générer trois versions du même path (clean / additive noise / one-sided noise) à partir d'un seed fixé.
**Durée estimée :** 1 session.

---

## Étape 4 — Estimateurs jump-robust

**Quand :** après Étape 3 validée.
**Quoi :** Step 3 du CLAUDE.md complet, dans l'ordre indiqué : `bipower.py` → `medrv_minrv.py` → `threshold.py` → `spot.py` → `preavg.py`.
**Sortie attendue :** cinq modules d'estimation, leurs tests de convergence sous no-jump et de biais sous jumps, et `01_estimator_comparison.ipynb` qui reproduit Table 1 d'Andersen-Dobrev-Schaumburg 2012 dans la mesure du possible.
**Critère pour passer à l'étape suivante :** convergence empirique vers la vraie variance intégrée à fréquence croissante, MedRV/MinRV moins de 10% de biais sous jumps modérés, BV non-truncated visiblement biaisé (négatif control). La fenêtre de spot exclut bien le point testé (assertion explicite dans les tests).
**Durée estimée :** 2 à 3 sessions (une par sous-step pour `medrv_minrv` et `threshold` qui demandent une lecture de papier).

---

## Étape 5 — Construction des e-values et tests d'e-power

**Quand :** après Étape 4 validée. **Pas avant.** Les e-values dépendent de σ̂, donc tant que les estimateurs ne sont pas solides, l'e-value ne peut pas être évalué.
**Quoi :** Step 4 du CLAUDE.md complet : `construct.py` → `mixture.py` → `eprocess.py`, plus `experiments/01_evalue_validity.py`.
**Sortie attendue :** trois modules dans `evalues/`, et surtout les quatre tests d'e-power du Step 4 :
1. distribution H0 vs H1,
2. e-power monotone en taille de saut,
3. dominance contre baselines triviaux (E≡1 et p-to-e calibré),
4. validité E[E_i] ≤ 1 sous H0 comme sanity check intégrée.

`01_evalue_validity.py` produit les trois figures cibles (séparation des distributions, courbe d'e-power, comparaison vs p-to-e).
**Critère pour passer à l'étape suivante :** séparation visuelle nette des distributions, e-power strictement positif et monotone croissant, dominance ≥ 0 vs p-to-e calibré sur au moins deux régimes de volatilité.
**Durée estimée :** 2 à 3 sessions. **C'est le cœur méthodologique du projet — ne pas brusquer.**

---

## Étape 6 — FDR algorithms, partie validation individuelle

**Quand :** après Étape 5 validée.
**Quoi :** Step 5 du CLAUDE.md, sous-steps 1 → 7 dans l'ordre. Implémenter les sept procédures, mais à cette étape ne faire que les **deux premiers tests** : self-validity sous H0 et deterministic stream cross-check.
**Sortie attendue :** sept modules dans `fdr/`, chacun avec ses tests de validité H0 (FDR ≤ α + tolérance MC) et la comparaison sur stream déterministe hand-verifiable.
**Critère pour passer à l'étape suivante :** chaque algorithme contrôle le FDR au niveau cible sur ses propres hypothèses de dépendance ; tous les algorithmes s'accordent sur le stream déterministe.
**Durée estimée :** 3 à 4 sessions. Donner le PDF du papier correspondant à chaque session (e-GAI, stopped e-BH, e-closure).

---

## Étape 7 — Pipeline orchestrator et métriques

**Quand :** après Étape 6 validée.
**Quoi :** Step 6 du CLAUDE.md : `pipeline/detector.py` et `pipeline/metrics.py`.
**Sortie attendue :** un orchestrateur unique qui prend `(simulator_config, estimator, fdr_algo)` et retourne `(rejection_set, metrics)`. Métriques implémentées : FDR empirique, puissance, détection delay, F1 sur localisation, wall-clock.
**Critère pour passer à l'étape suivante :** un end-to-end sur `grid_quick.yaml` (M=50 reps, une seule fréquence, une seule taille de saut) tourne en moins de 5 minutes et produit un parquet de résultats parsable.
**Durée estimée :** 1 à 2 sessions.

---

## Étape 8 — Grille Monte Carlo complète et comparaison supervisée

**Quand :** après Étape 7 validée. **Vérifie d'abord que ton compute peut tenir la grille** (estimer le temps total avec `grid_quick` × facteur d'agrandissement).
**Quoi :** Step 7 du CLAUDE.md. Lancer `experiments/02_power_grid.py` avec `grid_main.yaml`, qui inclut la dimension `wealth_fractions`. Puis générer les figures dans `03_results_figures.ipynb`. C'est ici qu'on fait la **comparaison head-to-head supervisée** (3ème test du Step 5).
**Sortie attendue :** un parquet de résultats complet, un tableau de ranking par cellule (frequency × jump regime × α × wealth fraction), et un set de figures : courbes FDR empirique vs α nominal, courbes puissance, comparaison wall-clock, effet du wealth fraction sur α-death.
**Critère pour passer à l'étape suivante :** FDR empirique ≤ α + 2·SE partout (validation finale), e-closure ≥ e-LOND en puissance d'au moins 10% à FDR=0.1 sur le régime modéré 5s, ranking clair.
**Durée estimée :** 1 session de lancement + une nuit de compute + 1 session de génération de figures.

---

## Étape 9 — Application données réelles et comparaison non-supervisée

**Quand :** après Étape 8 validée. **Plan B (TAQ) par défaut, Plan A (BMLL) si dispo, Plan C (LOBSTER) en backup.**
**Quoi :** Step 8 du CLAUDE.md : `experiments/04_real_data.py` avec la granularité d'input spécifiée (tick-by-tick, downsampling en code, cleaning conforme). C'est ici qu'on fait la **comparaison head-to-head non-supervisée** (4ème test du Step 5) : matrice d'accord entre procédures, stabilité par sous-échantillonnage, alignement avec annonces macro.
**Sortie attendue :** un tableau par stock × jour × algorithme du nombre de jumps détectés, une matrice d'accord Jaccard inter-procédures, un graphique d'alignement temporel avec les timestamps FOMC / earnings.
**Critère pour passer à l'étape suivante :** au minimum un jour × un asset traité end-to-end (smoke test) ; idéalement un mois × 5-10 assets pour la validation empirique.
**Durée estimée :** 2 à 3 sessions (la moitié est passée sur le data cleaning et la conformité de la granularité). Possiblement plus si le data acquisition prend du temps.

---

## Étape 10 — Reproductibilité finale et figures

**Quand :** dernière phase.
**Quoi :** s'assurer que `notebooks/03_results_figures.ipynb` regénère toutes les figures en un seul `jupyter nbconvert`, geler les seeds, snapshot du parquet final dans `data/processed/`, README.md propre, badges CI sur GitHub.
**Sortie attendue :** repo dans un état où n'importe qui peut cloner, installer, lancer un script et reproduire chaque figure de bout en bout à partir de seeds fixés.
**Critère de fin de projet :** tous les acceptance criteria du §5 du CLAUDE.md sont cochés.
**Durée estimée :** 1 à 2 sessions.

---

## Récapitulatif de séquencement

```
1. Bootstrap       ─→ Étape 1
2. Simulation      ─→ Étapes 2, 3
3. Estimateurs     ─→ Étape 4
4. E-values        ─→ Étape 5         ← cœur méthodologique
5. FDR algos       ─→ Étape 6
6. Pipeline        ─→ Étape 7
7. Expériences     ─→ Étapes 8, 9     ← cœur empirique
8. Wrap-up         ─→ Étape 10
```

Pas de saut autorisé : chaque étape consomme ce que la précédente produit. Les seules parallélisations possibles sont :
- au sein de l'Étape 4, les sous-modules d'estimateurs sont indépendants une fois `bipower.py` fait,
- au sein de l'Étape 6, les sept algorithmes FDR sont indépendants une fois la signature de fonction fixée,
- l'Étape 10 (polissage repo, badges, README) peut commencer en parallèle de l'Étape 9.

Si vous êtes en binôme, le découpage naturel est : binôme 1 sur Étapes 2-3-4-5, binôme 2 sur Étapes 6-7, puis les deux ensemble sur Étapes 8-9-10.

---

*Plan créé le 15 mai 2026. À mettre à jour seulement si un Step du CLAUDE.md change — l'ordre lui-même est stable.*
