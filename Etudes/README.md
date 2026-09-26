# Études expérimentales — Pipeline V5

Ce dossier regroupe les six études d'analyse menées sur le pipeline de dépistage opportuniste du risque cardiovasculaire à partir de radiographies thoraciques.
Toutes les études portent sur les résultats d'inférence du pipeline V5 appliqué à **501 images NIH ChestX-ray14**.

> **PFE** — *Trustworthy AI for Opportunistic Screening of Cardiovascular Risk from Chest X-Ray Images*
> Abdallah OUAHMANE — Faculté des Sciences de Rabat, Master MIT/TAM — Encadrant : Pr. Mohamed El Hassouni

---

## Structure du dossier

```
Etudes/
├── Models/
│   ├── unet_cardio_robust_v2_best .pth   # M1 — U-Net segmentation cardiaque
│   ├── unet_calcif_best_kagg.pth         # M2 — U-Net segmentation calcique
│   ├── calcif_model_final.pth            # M3 — EfficientNet-B0 classifieur calcique
│   └── cardio_risk_v2_best.pth           # M4 — EfficientNet-B0 risque cardiovasculaire
├── data/
│   └── resultats_phase4_v5.csv           # Résultats d'inférence du pipeline V5 (501 images)
├── etude1.ipynb                          # Ablation des modules
├── etude2.ipynb                          # Sources du signal calcique (M2 vs M3)
├── etude2b.ipynb                         # M2 en composante additionnelle + genèse de V5
├── etude3.ipynb                          # Sensibilité aux pondérations
├── etude4.ipynb                          # Gating par la confiance et calibration
├── etude5.6.ipynb                        # Comparaison end-to-end + validation des poids
└── README.md
```

**Point important :** les études n'exécutent aucun modèle. Elles lisent uniquement le fichier `resultats_phase4_v5.csv`. Le dossier `Models/` contient les checkpoints ayant servi à produire ce CSV lors de l'inférence (voir le notebook `Interferences/interference.ipynb` à la racine du dépôt) ; il permet de régénérer le CSV, mais n'est pas requis pour relancer les études. Aucun GPU n'est nécessaire.

> `calcif_model_final.pth` est le checkpoint M3 correct (chargement `strict=True`, epoch 20). Ne pas le confondre avec un éventuel `calcif_model_best.pth` d'une exécution antérieure, qui correspondait à un EfficientNet-B0 ImageNet non entraîné.

---

## Formule évaluée — Pipeline V5 (déployé)

```
risk_global = (0.15 × s_ctr + 0.40 × s_cong + 0.10 × s_calcif + 0.35 × prob_cardio) × 100
```

| Composante    | Source                              | Poids |
|---------------|-------------------------------------|-------|
| `s_ctr`       | M1 — U-Net segmentation cardiaque (CTR) | 0.15 |
| `s_cong`      | M4 — probabilité d'œdème            | 0.40  |
| `s_calcif`    | M3 — classifieur calcique EfficientNet-B0 | 0.10 |
| `prob_cardio` | M4 — probabilité de cardiomégalie   | 0.35  |

Performances de référence (n = 501) : **AUC 0.8916**, sensibilité 0.928, spécificité 0.766, VPP 0.888.

Vérité terrain utilisée dans toutes les études : une image est **positive** si son étiquette NIH contient `Cardiomegaly` ou `Edema`, **négative** sinon.

---

## Description des études

### Étude 1 — Ablation des modules (`etude1.ipynb`)
Quantifie l'apport de chaque module et compare les formules de fusion successives.
- AUC de chaque composante prise isolément (M1, M3, M4 œdème, M4 cardio).
- Ablations : V5 privée d'un module, poids restants renormalisés.
- Évolution des formules : uniforme (50/25/25) → V3 → V4 → V5.
- AUC par pathologie (cardiomégalie / œdème) et scores moyens par groupe clinique.
- Contrôle de cohérence : la formule recalculée est comparée à la colonne `risk_global` du CSV.

**Sortie :** `etude1_ablation_v5.png / .svg`

### Étude 2 — Sources du signal calcique (`etude2.ipynb`)
Quelle source calcique est la plus discriminante : le U-Net M2 (Dice 0.784) ou le classifieur M3 (F1 0.369) ?
- La composante calcique de V5 est remplacée par chaque source candidate, les trois autres poids restant fixes.
- Variantes de M2 testées : linéaire, racine, binaire (seuil 0.5 %), combinaisons M2+M3, absence de signal.
- Analyse du signal par groupe clinique pour expliquer le faible pouvoir discriminant de M2.

**Sortie :** `etude2_sources_calciques.png / .svg`

### Étude 2B — M2 en composante additionnelle (`etude2b.ipynb`)
M2 apporte-t-il une information complémentaire à M3, en s'y ajoutant plutôt qu'en le remplaçant ?
- Balayage du poids accordé à M2 (0 à 0.30), les quatre poids de V5 étant réduits proportionnellement.
- Seuil de significativité : ΔAUC > 0.005.
- **Genèse de V5** : balayage de `prob_cardio` à partir de V4, montrant la correction de l'anti-corrélation de V4 avec la cardiomégalie.
- AUC par pathologie pour V3, V4 et V5.

**Sortie :** `etude2b_m2_additionnel.png / .svg`

### Étude 3 — Sensibilité aux pondérations (`etude3.ipynb`)
La performance dépend-elle d'un réglage fin des quatre poids ?
1. **Balayage univarié** : chaque poids varie seul ; plage de stabilité et amplitude d'AUC par composante.
2. **Perturbation aléatoire** : 1 000 tirages autour de V5 (écart-type 0.05).
3. **Grille contrainte** sous planchers d'auditabilité (w_ctr ≥ 0.15, w_cong ≥ 0.15, w_calcif ≥ 0.10, w_cardio ≤ 0.40) : rang de V5 parmi les configurations admissibles.
4. **Bootstrap** (200 rééchantillonnages) : intervalle de confiance de l'AUC.

**Sortie :** `etude3_sensibilite_ponderations.png / .svg`

### Étude 4 — Gating par la confiance (`etude4.ipynb`)
Évalue le mécanisme *Confidence-Gated Fusion* : `s(τ) = s × 1[max(softmax) ≥ τ]`.
- Distribution de la confiance de M3 et M4.
- La confiance est-elle informative ? AUC de la confiance à prédire la justesse, diagramme de fiabilité et ECE de M4.
- Balayage de τ (0.20 → 0.95) pour M3 et pour M4.
- Résultat négatif documenté : il motive une perspective de recalibrage (température, Platt).

**Sortie :** `etude4_gating_confiance.png / .svg`

### Études 5 et 6 — Comparaison end-to-end et validation (`etude5.6.ipynb`)
**Étude 5** — Le pipeline modulaire bat-il un classifieur unique ?
- Comparaison de V5 au Module 4 seul (`1 − prob_normal`), avec IC 95 % par bootstrap stratifié (2 000 tirages).
- Tests de DeLong entre V5 et chaque formule de référence.
- Apport des composantes anatomiques (CTR, M3), y compris par régression logistique en validation croisée à 5 plis.

**Étude 6** — Les poids de V5 résistent-ils à la validation ?
- Recherche exhaustive sous contrainte d'auditabilité.
- Validation par 200 demi-échantillons stratifiés : optimisation sur une moitié, évaluation sur l'autre.
- Estimation du surajustement et AUC « honnête » à rapporter.
- Contribution de M3 sous validation indépendante (DeLong + distribution sur 200 partitions).

**Sortie :** `etudes_5_6_v5.png / .svg`

---

## Dictionnaire du fichier `resultats_phase4_v5.csv`

| Colonne | Description |
|---------|-------------|
| `image` | Nom du fichier image NIH |
| `label_reel` | Étiquette NIH (ex. `No Finding`, `Cardiomegaly`, `Edema`) |
| `ctr` | Rapport cardio-thoracique mesuré par M1 |
| `ctr_valide` | CTR plausible, dans l'intervalle [0.30 ; 0.75] |
| `cardio_flag` | Indicateur de cardiomégalie selon le CTR |
| `heart_cm`, `thorax_cm` | Largeurs cardiaque et thoracique estimées |
| `calcif_pct` | Surface calcifiée segmentée par M2 (%) |
| `calcif_label` | Classe calcique prédite par M3 |
| `confiance_m3` | Confiance de M3 (max du softmax) |
| `m3_utilise` | M3 pris en compte dans le score |
| `calcif_score` | Score calcique intermédiaire |
| `prob_normal`, `prob_cardio`, `prob_edema` | Probabilités softmax de M4 |
| `s_ctr`, `s_calcif`, `s_cong` | Composantes normalisées entrant dans la fusion |
| `risk_global` | Score de risque V5 (0–100) |
| `risk_label`, `niveau` | Niveau de risque attribué |

---

## Exécution

**Dépendances :**
```bash
pip install pandas numpy matplotlib scikit-learn scipy
```

**Chemin du CSV :** les notebooks ont été écrits pour Kaggle et pointent encore vers :
```python
CSV_PATH = '/kaggle/input/datasets/abdallahouahmane/resultas-phase4/resultats_phase4_v5.csv'
```
Pour une exécution locale depuis ce dossier, remplacer cette ligne par :
```python
CSV_PATH = 'data/resultats_phase4_v5.csv'
```
Pour charger le CSV directement depuis GitHub sans cloner le dépôt, utiliser l'URL **raw** (et non le lien `/blob/`, qui pointe vers la page HTML de visualisation) :
```python
CSV_PATH = 'https://raw.githubusercontent.com/sw-ouahmane/Projet-de-fin-d-etude/main/Etudes/data/resultats_phase4_v5.csv'
```

Les notebooks sont indépendants et peuvent être exécutés dans n'importe quel ordre. L'ordre numérique suit toutefois la logique de l'argumentation : contribution des modules → choix de la source calcique → robustesse des poids → gating → validation finale.

Chaque étude enregistre sa figure en PNG et SVG (dpi = 300), prête à intégrer dans le rapport.

---

## Limites

- L'évaluation porte sur les étiquettes NIH extraites par traitement du langage naturel : l'AUC mesure la concordance avec une annotation d'imagerie, non la survenue d'événements cardiovasculaires.
- Les poids de V5 ont été optimisés sur les mêmes 501 images que celles de l'évaluation ; l'Étude 6 quantifie le surajustement qui en résulte.
- M2 n'entre pas dans le score (Études 2 et 2B) ; il est conservé pour la visualisation des zones calcifiées.
