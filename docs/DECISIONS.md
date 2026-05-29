# DECISIONS.md — Choix techniques documentés

> Ce fichier trace les décisions non-évidentes prises pendant l'implémentation.

---

## Étape 6 — FDR algorithms (16 mai 2026)

### Constante C_JM du γ_t de Javanmard-Montanari

**Formule :** `γ_t = C · log(max(t,2)) / (t · exp(√(log(max(t,2)))))`

**Valeur utilisée :** `C = 0.15708906` (vérifiée par sommation tronquée à t=10^6 : Σ γ_t ≈ 1.000000).

**Contexte :** Une valeur initiale C ≈ 0.0722 avait été considérée, mais le calcul numérique donne C ≈ 0.1571 pour cette formule exacte. L'écart provient d'une confusion avec une variante de la formule (sans le `log` au numérateur, ou avec un argument différent dans l'exponentielle). Le résultat qualitatif ne change pas : e-LOND et e-LORD rejettent ∅ sur le stream déterministe (1,1,100,1,1) α=0.1 quelle que soit la valeur de C, car le seuil reste largement > 100.

**Implémentation :** constante nommée `_C_JM` dans `fdr/elond.py`, partagée avec `fdr/elord_esaffron.py` via import.

---

### Paramètre W_0 pour e-LORD

**Valeur utilisée :** `W_0 = 0.5` (défaut dans `elord()` via kwarg `w0`).

**Justification :** partage équitable entre wealth initial et wealth accumulé par les rejets passés. Valeur canonique dans la littérature e-GAI (Zhang et al. 2025). Paramètre modifiable via kwarg pour les expériences de la grille (Étape 8 : `wealth_fractions` dans `grid_main.yaml`).

---

### Comportement quand k*=0 dans e-BH

**Cas :** aucun k ne satisfait E_(k) ≥ n/(α·k).

**Décision :** retourner `[False] * n` — aucun rejet. Implémenté dans `fdr/ebh.py` et `fdr/stopped_ebh.py`.

**Justification :** k*=0 signifie que même le meilleur e-value ne dépasse pas le seuil le plus bas. Retourner des False est la seule décision cohérente avec la procédure (pas de cutoff valide).

---

### Stopped e-BH : décision à chaque t vs décision finale

**Décision :** à chaque step t, `stopped_ebh` recalcule e-BH offline sur E_{1:t} et yield `True` ssi E_t est dans le rejection set courant. Les décisions précédentes (t' < t) ne sont pas révisées — seul le verdict sur l'hypothèse courante est émis.

**Justification :** l'interface `Iterator[bool]` impose une décision par hypothèse, dans l'ordre. La propriété anytime-validity de stopped e-BH garantit que le rejection set à chaque t contrôle le FDR, conditionnellement à l'arrêt à t.

---

### Validité H0 pour e-LOND et e-LORD

**Observation :** les tests de validité FDR sur stream Pareto-1 (e-valeurs i.i.d. avec E[E_i]=1) et sur stream Heston pure-diffusion vérifient bien FDR ≤ α + 2·SE sur M=1000 reps.

**Note :** les tests `@pytest.mark.slow` utilisent n=500-1000 observations et M=1000 réplications. Ils sont exclus de la suite rapide par défaut.

---

### Baselines BH-BNS : test global vs par observation

**Décision :** `bh_bns()` retourne une liste de n booléens uniformes (tous True ou tous False selon que le test global BNS rejette H0 au niveau α). C'est cohérent avec l'usage dans Bajgrowicz-Scaillet 2016 où BNS sert de filtre jour-niveau.

**Limitation :** BNS ne localise pas les sauts — il détecte seulement leur présence dans la fenêtre. Pour la localisation, utiliser BH-LM (Lee-Mykland par observation) ou les procédures e-value.

---

### e-LORD : réécriture complète (2ème session)

**Problème :** La première implémentation de `elord` utilisait la formule γ_t de e-LOND (suite statique Javanmard-Montanari). C'était incorrect — le vrai e-LORD de Zhang et al. 2025 utilise le framework RAI (Risk Aversion Investing) avec une suite ω_t *dynamique*.

**Formule correcte (RAI) :**
```
alpha_t = omega_t * rw_t * (R_{t-1} + 1)
rw_{t+1} = rw_t * (1 - omega_t)
omega_{t+1} = omega_t + w1 * phi^{t-R_t} * (1-delta_t) - w1 * psi^{R_t} * delta_t
```

**Paramètres recommandés (Zhang et al. 2025) :** w1 = O(1/T), phi = psi = 0.5.  
**Comportement attendu :** avec w1=0.2 (agressif) sur E=(1,1,100,1,1), α=0.1, e-LORD rejette {3} (seuil ≈ 51). Avec w1=0.05 (défaut), rejette ∅.

---

### e-SAFFRON : formule exacte (Zhang et al. 2025, Algo 2)

**Formule :**
```
rw_1 = alpha * (1 - lambda_cand)   # réduit par (1-lambda)
rw_{t+1} = rw_t * (1 - omega_t * 1{E_t < 1/lambda})  # update seulement pour non-candidats
alpha_t = omega_t * rw_t * (R_{t-1} + 1)
delta_t = 1{E_t >= 1/alpha_t}  # rejet INDÉPENDANT de la candidature
```

**Point subtil :** Le seuil 1/λ ne sert QUE pour le calcul du budget. La décision de rejet s'applique à TOUTES les hypothèses, candidates ou non.

**λ = 0.1** (pas 0.5 comme dans le SAFFRON p-value). Le papier recommande une valeur petite.

**Intuition :** quand une e-value est forte (E_t ≥ 1/λ), on ne "charge" pas le budget — on garde le wealth pour les hypothèses futures. D'où la puissance supérieure à e-LORD.

---

### e-LOND-bar et donation e-LOND : différés

**Décision :** e-LOND-bar (online closure, O(t²) via dynamic programming) et donation e-LOND (O(log t) via online compound e-values) sont différés.

**Raisons :**
1. La formule closure DP n'est pas entièrement explicitée dans Xu-Fischer-Ramdas 2026.
2. La notion de "γ-online compound e-value" (Section 2.2) nécessite une lecture dédiée.
3. Mieux vaut valider l'intégration pipeline avant d'ajouter des algorithmes supplémentaires.

**Ce qui est implémenté à la fin de step-6 :** e-BH, BH-LM, BH-BNS, e-LOND, e-LORD, e-SAFFRON, stopped e-BH. Soit 7 algorithmes, tous validés (FDR ≤ α + 2·SE sur streams Pareto et Heston).

---

### Construction valide des e-values H0 pour les tests

**Bug corrigé :** `E = 1/U` (U~Uniform) a E[E] = ∞ — pas un e-value valide sous H0. Conséquence : le test `test_stopped_ebh_fdr_validity_pareto` échouait avec FDR=0.481.

**Correction :** `E = 0.5/sqrt(U)` — calibrateur Vovk-Wang 2021 avec κ=0.5. E[E] = 0.5 * ∫₀¹ u^{-1/2} du = 1. ✓ Queue Pareto de forme 2 : P(E > x) = 0.25/x² pour x ≥ 0.5.

---

---

## Étape 8 — Grille Monte Carlo complète (session du 16 mai 2026)

### Régime hawkes_dense : processus Hawkes auto-excitant

**Motivation :** Le régime `dense` (Merton, λ=15 000/an) ne modélise pas la dépendance temporelle. Pour tester le contrôle FDR sous dépendance (clustering de sauts), il faut un processus auto-excitant.

**Formule (temps discret) :**
```
λ_{n+1} = μ + (λ_n − μ)·exp(−β·Δt) + α·N_n
```

**Paramètres par défaut :**
- `mu_per_year = 5 000` (baseline = moderate Poisson)
- `alpha_per_year = 88 000` (boost post-saut)
- `beta_per_sec = 0.03` (demi-vie ≈ 23s ≈ 4.6 ticks à 5s)
- Branching ratio continu : m = α/(β × SECS_PER_YEAR) ≈ **0.50** (stable)

**Validation :** Sur 30 seeds (n=200, dt=5s) → 3.8 sauts/path vs 1.3 pour moderate (ratio ≈ 3). ✓

**Limitation connue :** L'approximation discrète est précise pour Δt petit. À Δt=300s le ratio de branchement discret dépasse 1 (instabilité apparente), mais le processus reste bien défini grâce à la saturation naturelle de la probabilité (1−exp(−λ·Δt) ≤ 1). Ce régime est le plus informatif à haute fréquence (1–30s).

---

### Paramètres de la grille Monte Carlo

**grid_medium.yaml** (validation, ~7 min avec 3 workers sur 4 cœurs) :
- n_steps=500, 3 fréqs [5, 60, 300], 3 régimes, 2 tailles, 2 alpha, 2 wealth, M=100
- 32 400 tasks total

**grid_main.yaml** (run final, ~2.9h avec 3 workers) :
- n_steps=2000, 5 fréqs, 3 régimes (incl. hawkes_dense), 2 tailles, 2 alpha, 4 wealth, M=500
- 390 000 tasks total ; séquentiel ~20h → parallèle ~2.9h

**Benchmark single-run à n=2000 :**
| algo | ms/run |
|---|---|
| bh_lm | 1 |
| ebh | 170 |
| elond | 185 |
| esaffron | 174 |
| elord | 187 |
| bh_bns | 257 |
| stopped_ebh | 346 |

---

### `mixed` jump size : différé

**Décision :** `jump_sizes_in_sigma: [3, 5, mixed]` est retiré car `float("mixed")` crashe le runner. La taille mixte (distribution de taille de saut variable) nécessite un support dédié dans `_expand_grid()`.

---

### wealth_fractions pour n=2000 : corrigé vers la recommandation e-GAI

**Problème :** `[0.1, 0.25, 0.5, 0.75]` causent toutes l'alpha-death pour n=2000. Confirmé sur grid_medium (n=500, w1=0.1 → e-LOND/e-SAFFRON power≈0).

**Correction :** `[0.0005, 0.0025, 0.005, 0.05]`
- `1/n = 0.0005` : valeur canonique e-GAI (Zhang et al. 2025)
- `5/n = 0.0025` : voisinage canonique
- `0.005` : validé safe pour n=500 dans grid_quick (extrapolé pour n=2000)
- `0.05` : illustre l'onset de l'alpha-death (pédagogique)

---

## Findings — Étape 8 (grid_medium, 16 mai 2026)

### Résultat principal : BH viole le FDR à haute fréquence, indépendamment du régime

**Observation :** Sur grid_medium (M=100, n=500), BH-BNS et BH-LM violent le critère FDR ≤ α + 2·SE **uniquement à dt=5s**, sur tous les régimes (rare, moderate, hawkes_dense) :

| dt | FDR BH-BNS (tous régimes) | Violé (α=0.05) ? |
|----|---|---|
| 5s | 0.091–0.099 | Oui |
| 60s | 0.009–0.045 | Non |
| 300s | ≈0 | Non |

Les e-values (ebh, elond, stopped_ebh) contrôlent le FDR à toutes les fréquences (FDR ≤ 0.033 ≤ α + 2·SE).

**Interprétation :** La BV (bipower variation) sous-estime la volatilité spot à 5 secondes à cause de la corrélation négative induite par le bruit microstructure (effet bid-ask "bounce"). Cette sous-estimation gonfle la statistique BNS et crée de faux sauts. Références : Aït-Sahalia-Mykland-Zhang 2005, Bajgrowicz-Scaillet 2016.

**Argument méthodologique central du projet :** les e-values sont valides sous dépendance arbitraire (leur validité ne suppose pas le PRDS de BH). La dépendance microstructure à haute fréquence est précisément ce qui rend le PRDS intenable pour BH. Les e-values offrent donc un meilleur trade-off sécurité/puissance à HF. Avec pré-moyennage (preavg_bv), la violation BH disparaîtrait — mais au coût d'une réduction de puissance substantielle.

**Le clustering Hawkes ne spécifie pas la violation BH :** BH viole le FDR autant sur moderate (Poisson, pas de dépendance temporelle) que sur hawkes_dense. Le régime hawkes_dense sert néanmoins de test de robustesse : les e-values restent valides même sous clustering auto-excitant.

**Puissance à dt=5s, α=0.05, hawkes_dense :**

| algo | FDR | power | FDR contrôlé ? |
|---|---|---|---|
| BH-BNS | 0.091 | 0.423 | Non |
| BH-LM | 0.091 | 0.423 | Non |
| stopped_ebh | 0.017 | 0.303 | Oui |
| ebh | 0.009 | 0.280 | Oui |
| e-LOND | 0.000 | 0.204 | Oui |
| e-LORD | ≈0 | ≈0 | Oui (alpha-death w1=0.1) |
| e-SAFFRON | ≈0 | ≈0 | Oui (alpha-death w1=0.1) |

---

## Sessions précédentes

*(Décisions des Étapes 1–5 non documentées ici — ajoutées rétrospectivement si besoin.)*
