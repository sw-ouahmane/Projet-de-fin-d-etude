# Inférence — Pipeline V5

Ce dossier contient le notebook qui exécute le pipeline complet (M1 → M4) sur un tirage stratifié d'images NIH ChestX-ray14 et produit le fichier de résultats consommé par les six études du dossier `Etudes/`.

> **PFE** — *Trustworthy AI for Opportunistic Screening of Cardiovascular Risk from Chest X-Ray Images*
> Abdallah OUAHMANE — FSR Rabat — Encadrant : Pr. Mohamed El Hassouni

---

## Structure du dossier

```
Interferences/
├── interference.ipynb        # Pipeline d'inférence complet
└── README.md
```

**Entrées :** les checkpoints `.pth` des quatre modules (dossier `Etudes/Models/` du dépôt) et les images + métadonnées NIH ChestX-ray14 (CLAHE).
**Sorties :** `resultats_phase4_v5.csv` (501 lignes — c'est le fichier utilisé par `Etudes/data/resultats_phase4_v5.csv`) et `pipeline_v5_metriques.png / .svg`.

---

## Ce que fait le notebook

1. **Chargement des quatre modèles**
   - **M1** — U-Net segmentation cardiaque (chargement strict).
   - **M2** — U-Net segmentation calcique, conservé pour la visualisation (hors fusion).
   - **M3** — EfficientNet-B0 classifieur calcique (`calcif_model_final.pth`, chargement strict), conservé pour traçabilité mais exclu de la fusion.
   - **M4** — EfficientNet-B0 risque cardiovasculaire (3 classes : Normal / Cardiomegaly / Edema).

2. **Constitution du jeu de test** — tirage stratifié de 167 images par classe (Normal, Cardiomegaly seule, Edema seule), seed = 99, mélangé avec seed = 42 → 501 images au total.
   > Limite documentée dans le notebook : ce tirage ne garantit pas l'exclusion des images ayant servi à l'entraînement de M4 ; un recouvrement est possible.

3. **Analyse par image** (`analyser_image`)
   - Calcul du **CTR** à partir du masque M1 : contour cardiaque le plus grand centré dans la zone anatomique attendue, diamètre thoracique approximé à 75 % de la largeur image, conversion en cm via le pixel spacing corrigé, correction de −0.05 en incidence AP.
   - **Plausibilité du CTR** : `ctr_valide` si CTR ∈ [0.30 ; 0.75] ; sinon le niveau de risque final est forcé à `INDÉTERMINÉ`.
   - Surface calcifiée (`calcif_pct`) depuis le masque M2, à titre de visualisation.
   - Probabilités et confiance de M3 (conservées mais non injectées dans le score).
   - Probabilités softmax de M4 (`prob_normal`, `prob_cardio`, `prob_edema`).
   - Normalisation des composantes : `s_ctr = clip((ctr − 0.40) / 0.20, 0, 1)`, `s_calcif = clip(prob_calcification + prob_aortic_enlargement, 0, 1)`, `s_cong = prob_edema`.
   - **Fusion V5** : `risk_global = (0.15×s_ctr + 0.40×s_cong + 0.10×s_calcif + 0.35×prob_cardio) × 100`.
   - Catégorisation : `FAIBLE` (< 25 %), `MODÉRÉ` (25–35 %), `ÉLEVÉ` (≥ 35 %, seuil maximisant l'indice de Youden), `INDÉTERMINÉ` si le CTR n'est pas plausible.

4. **Métriques** — matrice de confusion, sensibilité, spécificité, VPP, VPN, exactitude, AUC-ROC, seuil de Youden recalculé, répartition des niveaux de risque.

5. **Contrôle de reproductibilité** — compare les colonnes brutes (`ctr`, `s_ctr`, `s_cong`, `prob_edema`, `prob_normal`) à un CSV antérieur pour vérifier que seule la formule de fusion a changé entre les versions du pipeline.

6. **Figure** — courbe ROC, matrice de confusion, distribution des scores par classe clinique (`pipeline_v5_metriques.png / .svg`).

---

## À corriger avant une exécution locale

Le notebook a été écrit pour Kaggle et référence des chemins et des noms de fichiers spécifiques à cet environnement, différents de ceux du dépôt :

| Variable | Valeur dans le notebook (Kaggle) | Fichier réel dans `Etudes/Models/` |
|---|---|---|
| `MODELS_DIR` | `/kaggle/input/datasets/abdallahouahmane/pfe-cardio-models` | `Etudes/Models/` |
| `UNET_CARDIO` | `unet_cardio_robust_v2_best.pth` | `unet_cardio_robust_v2_best .pth` *(espace avant l'extension)* |
| `UNET_CALCIF` | `unet_calcif_best_kagg (2).pth` | `unet_calcif_best_kagg.pth` |
| `CALCIF_MODEL` | `calcif_model_final.pth` | `calcif_model_final.pth` ✓ identique |
| `RISK_MODEL` | `cardio_risk_v2_best (1).pth` | `cardio_risk_v2_best.pth` |
| `CSV_PATH` / `IMG_DIR` | dataset Kaggle `rahulogoel/clahe-enhancement-on-chestx-ray14` | à fournir localement (images + `Data_Entry.csv` NIH, version CLAHE) |

Les suffixes `(1)` / `(2)` dans les noms Kaggle viennent de doublons d'upload et n'existent pas dans les fichiers du dépôt. Adapter ces chemins avant toute exécution hors Kaggle.

> Deux détails historiques du script, sans impact sur les résultats : l'en-tête du docstring indique « PIPELINE v5 » mais les messages console imprimés à l'exécution disent encore « PIPELINE V3 » ; de même, le commentaire au-dessus du chargement de M3 décrit un ancien scénario de chargement `strict=False`, alors que le code appelle `load_state_dict` sans cet argument (donc en mode strict par défaut), conformément au bon checkpoint `calcif_model_final.pth`.

---

## Dépendances

```bash
pip install torch torchvision opencv-python pandas numpy matplotlib seaborn scikit-learn pillow
```

GPU recommandé (bascule automatique CPU/GPU via `torch.cuda.is_available()`), mais non requis pour un test sur un petit sous-ensemble d'images.

---

## Sortie produite

Le fichier `resultats_phase4_v5.csv` généré ici est celui consommé tel quel par les six études du dossier [`Etudes/`](../Etudes/README.md) : sa structure de colonnes (`ctr`, `s_ctr`, `s_calcif`, `s_cong`, `prob_cardio`, `risk_global`, etc.) est documentée dans le README de ce dossier.
