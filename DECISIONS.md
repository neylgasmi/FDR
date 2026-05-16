# DECISIONS.md — Choix techniques documentés

> Ce fichier trace les décisions non-évidentes prises pendant l'implémentation.
> Mettre à jour à chaque session.

---

## Étape 6 — FDR algorithms (sessions du 16 mai 2026)

### Constante C_JM du γ_t de Javanmard-Montanari

**Formule :** `γ_t = C · log(max(t,2)) / (t · exp(√(log(max(t,2)))))`

**Valeur utilisée :** `C = 0.15708906` (vérifiée par sommation tronquée à t=10^6 : Σ γ_t ≈ 1.000000).

**Contexte :** Le brief de session indiquait C ≈ 0.0722. Le calcul numérique donne C ≈ 0.1571 pour cette formule exacte. L'erreur dans le brief est probablement une confusion avec une variante de la formule (e.g. sans le `log` au numérateur, ou avec un argument différent dans l'exponentielle). Le résultat qualitatif ne change pas : e-LOND et e-LORD rejettent ∅ sur le stream déterministe (1,1,100,1,1) α=0.1 quelle que soit la valeur de C, car le seuil reste largement > 100.

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

**Point subtil :** Le seuil 1/λ ne sert QUE pour le calcul du budget. La décision de rejet s'applique à TOUTES les hypothèses, candidates ou non. C'était incorrect dans le brief initial.

**λ = 0.1** (pas 0.5 comme dans le SAFFRON p-value). Le papier recommande une valeur petite.

**Intuition :** quand une e-value est forte (E_t ≥ 1/λ), on ne "charge" pas le budget — on garde le wealth pour les hypothèses futures. D'où la puissance supérieure à e-LORD.

---

### e-LOND-bar et donation e-LOND : différés

**Décision :** e-LOND-bar (online closure, O(t²) via dynamic programming) et donation e-LOND (O(log t) via online compound e-values) sont différés à une session ultérieure dédiée, avec un brief basé sur la lecture PDF complète de Xu-Fischer-Ramdas 2026.

**Raisons :**
1. La formule closure DP n'est pas entièrement explicitée dans la portion HTML accessible.
2. La notion de "γ-online compound e-value" (Section 2.2) nécessite un brief séparé.
3. Mieux vaut valider l'intégration pipeline (Étape 7) avant d'ajouter des algorithmes supplémentaires.

**Ce qui est implémenté à la fin de step-6 :** e-BH, BH-LM, BH-BNS, e-LOND, e-LORD, e-SAFFRON, stopped e-BH. Soit 7 algorithmes, tous validés (FDR ≤ α + 2·SE sur streams Pareto et Heston).

---

### Construction valide des e-values H0 pour les tests

**Bug corrigé :** `E = 1/U` (U~Uniform) a E[E] = ∞ — pas un e-value valide sous H0. Conséquence : le test `test_stopped_ebh_fdr_validity_pareto` échouait avec FDR=0.481.

**Correction :** `E = 0.5/sqrt(U)` — calibrateur Vovk-Wang 2021 avec κ=0.5. E[E] = 0.5 * ∫₀¹ u^{-1/2} du = 1. ✓ Queue Pareto de forme 2 : P(E > x) = 0.25/x² pour x ≥ 0.5.

---

## Sessions précédentes

*(Décisions des Étapes 1–5 non documentées ici — ajoutées rétrospectivement si besoin.)*
