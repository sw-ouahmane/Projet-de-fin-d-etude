# Pipeline — Démonstration sur image individuelle

Ce dossier retrace l'évolution du pipeline de dépistage à travers trois scripts de démonstration, chacun analysant une ou plusieurs images une par une (masques, Grad-CAM, contributions au score) plutôt qu'un lot de 501 images comme dans `Interferences/`. C'est la version `pipline_v5.ipynb` qui correspond à la formule finale déployée et documentée dans `Etudes/`.

> **PFE** — *Trustworthy AI for Opportunistic Screening of Cardiovascular Risk from Chest X-Ray Images*
> Abdallah OUAHMANE — FSR Rabat — Encadrant : Pr. Mohamed El Hassouni

---

## Structure du dossier

```
pipline/
├── pipline_v1.ipynb   # Formule V3 (0.45×CTR + 0.55×congestion) — démonstration multi-images
├── pipline_v3.ipynb   # Fusion uniforme (0.50/0.25/0.25) + corrections de bugs d'ingénierie
├── pipline_v5.ipynb   # Formule V5 déployée (4 composantes) — version finale
└── README.md
```

**Point d'attention sur les noms de fichiers :** la numérotation des fichiers (`v1`, `v3`, `v5`) ne correspond pas à l'ordre chronologique réel des formules de fusion qu'ils contiennent. Le tableau ci-dessous clarifie ce que contient chaque fichier.

---

## Contenu de chaque notebook

| Fichier | Titre interne | Formule de fusion | Checkpoint M3 utilisé | Statut |
|---|---|---|---|---|
| `pipline_v1.ipynb` | « PIPELINE V3 — Démonstration » | `0.45×s_ctr + 0.55×s_cong` | `calcif_model_final.pth` (strict) | Formule V3, M2/M3 en visualisation seule |
| `pipline_v3.ipynb` | « Version corrigée [Bug #2+#5+#6] » | `0.50×s_ctr + 0.25×s_calcif + 0.25×s_cong` (fusion uniforme) | `calcif_model_best.pth` (`strict=False`) | **Utilise le checkpoint M3 incorrect** — voir avertissement ci-dessous |
| `pipline_v5.ipynb` | « PIPELINE V5 — Démonstration » | `0.15×s_ctr + 0.40×s_cong + 0.10×s_calcif + 0.35×prob_cardio` | `calcif_model_final.pth` (strict) | **Version finale déployée** |

### `pipline_v1.ipynb` — Formule V3
Applique la fusion à deux composantes retenue avant l'introduction de `prob_cardio` (Étude 2B). Rapporte en commentaire les résultats des Études 1 à 6 tels qu'observés pour cette formule (AUC 0.7523, IC 95 % [0.709 ; 0.796]). Charge les 4 modèles en mode strict et inclut un Grad-CAM basé sur les vrais gradients (`register_full_backward_hook`).

### `pipline_v3.ipynb` — Corrections d'ingénierie (fusion uniforme)
Ce script documente trois corrections de performance/fiabilité, indépendantes du choix de la formule :
- **Bug #2** — le CSV de métadonnées NIH (112k lignes) était relu à chaque image ; il est désormais chargé une seule fois et indexé (`df_meta`).
- **Bug #5** — absence d'avertissement quand le CTR n'est pas calibré faute de métadonnées (image externe sans entrée CSV) ; un message explicite (`ctr_warning`) est ajouté.
- **Bug #6** — les transforms `torchvision` (`T512`, `T224`) étaient recréés à chaque appel ; ils sont sortis en constantes globales.

> ⚠️ **Ce script charge `calcif_model_best.pth` avec `strict=False`.** D'après les notes techniques du projet, ce checkpoint est un EfficientNet-B0 ImageNet non entraîné sur la tâche calcique — un chargement `strict=False` masque silencieusement l'incompatibilité au lieu de lever une erreur. Le bon checkpoint est `calcif_model_final.pth`, utilisé (en mode strict) dans `pipline_v1.ipynb` et `pipline_v5.ipynb`. Ce fichier est conservé ici comme trace de ce diagnostic ; il ne doit pas servir de référence pour une nouvelle exécution.
> Le Grad-CAM de ce script pondère aussi les activations par leur propre moyenne spatiale plutôt que par celle du gradient rétropropagé (CAM, et non Grad-CAM au sens strict) — cohérent avec la distinction documentée dans le projet.

### `pipline_v5.ipynb` — Formule V5 déployée (version finale)
Charge les 4 modèles en mode strict, avec le commentaire explicite sur l'incident du checkpoint M3. Calcule et affiche, pour chaque image :
- **M1** (poids 15 %) — CTR, plausibilité anatomique (`INDÉTERMINÉ` si CTR ∉ [0.30 ; 0.75]).
- **M2** (visualisation, hors fusion) — surface calcifiée segmentée.
- **M3** (poids 10 %) — probabilités des 3 classes calciques, `s_calcif`.
- **M4** (poids 75 % au total : 40 % congestion + 35 % cardiomégalie) — probabilités des 3 classes de risque, Grad-CAM (vrais gradients), `s_cong` et `prob_cardio`.

Seuils de catégorisation : `FAIBLE` (< 25 %), `MODÉRÉ` (25–35 %), `ÉLEVÉ` (≥ 35 %), `INDÉTERMINÉ` (CTR implausible).

**Sortie par image :** figure 8 panneaux (`v5_<nom>.png / .svg`) — radiographie, masque M1 + CTR, masque M2, Grad-CAM M4, probabilités M4, probabilités M3, contributions au score. Un CSV récapitulatif (`resultats_demonstration_v5.csv`) est produit pour l'ensemble des images traitées dans l'exécution.

Résultats de référence rappelés en fin de script : AUC 0.8916 sur les 501 images (IC 95 % [0.8572 ; 0.9229]), AUC 0.8909 ± 0.0170 en validation indépendante (Étude 6), sensibilité 0.928 / spécificité 0.766 / VPP 0.888 au seuil de 35 %.

---

## Rôle des modules dans la formule V5

| Module | Rôle | Poids dans V5 |
|---|---|---|
| M1 — U-Net segmentation cardiaque | CTR → `s_ctr` | 0.15 |
| M2 — U-Net segmentation calcique | Visualisation uniquement (Étude 2B) | — |
| M3 — EfficientNet-B0 classifieur calcique | `s_calcif` | 0.10 |
| M4 — EfficientNet-B0 risque cardiovasculaire | `s_cong` (œdème) + `prob_cardio` (cardiomégalie) | 0.40 + 0.35 |

---

## À corriger avant une exécution locale

Comme pour `Interferences/interference.ipynb`, les trois scripts pointent vers des chemins et des noms de fichiers Kaggle :

- `MODELS_DIR` → remplacer par `Etudes/Models/`.
- `UNET_CARDIO` : les scripts utilisent `unet_cardio_robust_best.pth` (`v1`, `v3`) ou `unet_cardio_robust_v2_best.pth` (`v5`) ; le fichier réel dans le dépôt est `unet_cardio_robust_v2_best .pth` (espace avant l'extension).
- `UNET_CALCIF` : `unet_calcif_best_kagg (2).pth` dans les trois scripts → fichier réel `unet_calcif_best_kagg.pth`.
- `RISK_MODEL` : `cardio_risk_v2_best (1).pth` dans les trois scripts → fichier réel `cardio_risk_v2_best.pth`.
- `CALCIF_MODEL` : `calcif_model_final.pth` dans `v1` et `v5` (correct, identique au fichier du dépôt) ; **`calcif_model_best.pth` dans `v3`, qui n'existe pas dans `Etudes/Models/`** — ce fichier n'a jamais été retenu comme checkpoint définitif.
- `CSV_PATH` / `IMG_DIR` / `IMG_EXT` — jeux d'images NIH (CLAHE) et images externes de démonstration (`im1-Chest-X-ray.jpg`, `im2.png`, `im3.png`) non inclus dans ce dépôt ; à fournir séparément pour rejouer les démonstrations telles quelles, ou à remplacer par vos propres radiographies.

---

## Dépendances

```bash
pip install torch torchvision opencv-python pandas numpy matplotlib pillow
```

GPU recommandé mais non requis (bascule automatique CPU/GPU).

---

## Lien avec le reste du dépôt

- La formule et les seuils de `pipline_v5.ipynb` sont ceux évalués en masse dans [`Interferences/interference.ipynb`](../Interferences/README.md) sur 501 images, et analysés en détail dans les six études de [`Etudes/`](../Etudes/README.md).
- `pipline_v3.ipynb` documente un incident de checkpoint (M3) à ne pas reproduire ; le diagnostic complet de cet incident est discuté dans les notes techniques du projet, pas dans ce dossier.
