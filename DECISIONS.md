# DECISIONS.md — Choix techniques documentés

> Ce fichier trace les décisions non-évidentes prises pendant l'implémentation.
> Mettre à jour à chaque session.

---

## Étape 6 — FDR algorithms (session du 16 mai 2026)

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

## Sessions précédentes

*(Décisions des Étapes 1–5 non documentées ici — ajoutées rétrospectivement si besoin.)*
