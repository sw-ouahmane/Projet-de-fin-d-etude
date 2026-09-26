# Trustworthy AI for Opportunistic Screening of Cardiovascular Risk from Chest X-Ray Images

Projet de Fin d'Études (PFE) — Master MIT/TAM, Informatique et IA appliquée
**Faculté des Sciences de Rabat, Université Mohammed V**
Réalisé par **Abdallah OUAHMANE** — Encadrant : **Pr. Mohamed El Hassouni**

📄 Rapport complet : [`PFE_Cardiovascular_AI_v2.pdf`](./PFE_Cardiovascular_AI_v2.pdf)

---

## Objectif du projet

Construire un pipeline d'intelligence artificielle **auditable et cliniquement défendable** pour le dépistage opportuniste du risque cardiovasculaire à partir de radiographies thoraciques standard. Plutôt qu'un unique réseau de neurones opaque, le système combine **quatre modules spécialisés**, chacun mesurant un indicateur anatomique ou clinique reconnu (rapport cardio-thoracique, congestion pulmonaire, calcifications vasculaires), fusionnés par une formule pondérée dont chaque composante est justifiée expérimentalement — et non apprise en boîte noire.

L'ensemble du travail (entraînement, inférence, six études de validation, rapport) est disponible dans ce dépôt.

---

## Aperçu du pipeline

```
Radiographie thoracique
        │
        ├── M1 — U-Net segmentation cardiaque ──────► CTR (rapport cardio-thoracique)
        ├── M2 — U-Net segmentation calcique ───────► visualisation uniquement
        ├── M3 — EfficientNet-B0 classifieur calcique ─► s_calcif
        └── M4 — EfficientNet-B0 risque cardiovasculaire ─► s_cong (œdème), prob_cardio (cardiomégalie)
                        │
                        ▼
        risk_global = (0.15×s_ctr + 0.40×s_cong + 0.10×s_calcif + 0.35×prob_cardio) × 100
                        │
                        ▼
         FAIBLE (<25 %) · MODÉRÉ (25–35 %) · ÉLEVÉ (≥35 %) · INDÉTERMINÉ (CTR implausible)
```

**Performance du pipeline final (V5)**, sur 501 images NIH ChestX-ray14 : **AUC = 0.8916** [IC 95 % 0.857–0.923], sensibilité 0.928, spécificité 0.766, VPP 0.888. Validation indépendante (Étude 6) : AUC = 0.8909 ± 0.017.

---

## Structure du dépôt

| Dossier | Contenu | README |
|---|---|---|
| [`Models/`](./Models) | Notebooks d'entraînement des 4 modules (M1 à M4) | *(voir résumé ci-dessous)* |
| [`Etudes/`](./Etudes) | Checkpoints `.pth` entraînés, résultats d'inférence (CSV), 6 notebooks d'analyse et de validation | [README](./Etudes/README.md) |
| [`Interferences/`](./Interferences) | Notebook exécutant le pipeline complet sur 501 images et produisant le CSV de résultats | [README](./Interferences/README.md) |
| [`pipline/`](./pipline) | Notebooks de démonstration du pipeline sur une image individuelle (3 versions, évolution de la formule de fusion) | [README](./pipline/README.md) |
| [`datasets/`](./datasets) | Description des 3 jeux de données publics utilisés (non hébergés ici) | [README](./datasets/README.md) |
| `PFE_Cardiovascular_AI_v2.pdf` | Rapport complet du PFE | — |

---

## Les quatre modules

| Module | Rôle | Architecture | Entraînement | Performance |
|---|---|---|---|---|
| **[M1](./Models/M1_unet%20cardio%20robust.ipynb)** | Segmentation cardiaque → calcul du CTR | U-Net (4 niveaux) | JSRT (247 images) + VinDr-CXR (masques par vote majoritaire d'annotateurs), `DiceBCELoss`, Adam lr=1e-4 | **Dice 0.9580**, IoU 0.7472 |
| **[M2](./Models/M2_CALCIFICATIONS%20VASCULAIRES_segmenteur.ipynb)** | Segmentation des zones calcifiées (visualisation) | U-Net | VinBigData Chest X-ray | Dice 0.784 |
| **[M3](./Models/M3_CALCIFICATIONS%20VASCULAIRES%20_classificateur.ipynb)** | Classification calcique (Normal / Calcification / Aortic enlargement) | EfficientNet-B0 (entrée 1 canal) | VinBigData Chest X-ray | AUC 0.7037 |
| **[M4](./Models/M4_cardio_risk_v2.ipynb)** | Classification du risque (Normal / Cardiomegaly / Edema) | EfficientNet-B0 (entrée 1 canal) | NIH ChestX-ray14 CLAHE, jeu équilibré (2000/2000/2000 images) | **AUC 0.9094** (module dominant) |

Les checkpoints entraînés (`.pth`) sont disponibles dans [`Etudes/Models/`](./Etudes/Models).

> **M1** a fait l'objet d'une correction majeure documentée dans le rapport (§6) : augmentation appariée image/masque, retrait du retournement horizontal (anatomiquement invalide sur une radiographie), masques VinDr par vote majoritaire au lieu de l'union des annotations, et séparation stricte du test JSRT — ces corrections ont porté le Dice de 0.8547 à 0.9580.
> **M3** a également fait l'objet d'un diagnostic important : un ancien checkpoint (`calcif_model_best.pth`) chargeait silencieusement un EfficientNet-B0 non entraîné à cause d'un chargement `strict=False`. Le checkpoint correct est `calcif_model_final.pth`.

---

## De l'entraînement au score final

1. **Entraînement** (dossier `Models/`) — les quatre modules sont entraînés séparément sur trois jeux de données publics complémentaires (voir [`datasets/README.md`](./datasets/README.md) : VinBigData, NIH ChestX-ray14 CLAHE, JSRT).
2. **Inférence en masse** (dossier `Interferences/`) — le pipeline complet est exécuté sur 501 images NIH tirées de façon stratifiée, produisant `resultats_phase4_v5.csv`.
3. **Validation et études** (dossier `Etudes/`) — six études quantifient l'apport de chaque module, comparent les sources du signal calcique, testent la sensibilité et la stabilité des pondérations, évaluent un mécanisme de gating par la confiance, et valident le pipeline contre un classificateur unique et contre le surajustement.
4. **Démonstration** (dossier `pipline/`) — trois notebooks illustrent le pipeline sur une image à la fois, avec cartes de segmentation et Grad-CAM, retraçant l'évolution de la formule de fusion (V3 → fusion uniforme → V5).

---

## Formule de fusion — évolution

| Version | Formule | AUC (501 images) |
|---|---|---|
| Fusion uniforme (a priori) | `0.50×s_ctr + 0.25×s_calcif + 0.25×s_cong` | — |
| V3 | `0.45×s_ctr + 0.55×s_cong` | 0.7523 |
| V4 (optimum sans cardiomégalie) | `0.35×s_ctr + 0.45×s_cong + 0.20×s_calcif` | — |
| **V5 (déployée)** | `0.15×s_ctr + 0.40×s_cong + 0.10×s_calcif + 0.35×prob_cardio` | **0.8916** |

Le passage à V5 a corrigé une anti-corrélation de V4 avec la cardiomégalie (voir Étude 2B), ce qui explique le principal gain de performance.

---

## Jeux de données

Trois jeux de données publics sont utilisés (non hébergés dans ce dépôt en raison de leur taille) :

- **[NIH ChestX-ray14 (CLAHE)](https://www.kaggle.com/datasets/rahulogoel/clahe-enhancement-on-chestx-ray14)** — ~112 000 radiographies, 14 pathologies — entraînement de M4 et jeu d'évaluation du pipeline.
- **[VinDr-CXR / VinBigData](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)** — 18 000 radiographies annotées (boîtes englobantes) — entraînement de M1, M2, M3.
- **[JSRT](http://db.jsrt.or.jp/eng.php)** — 247 radiographies haute résolution — segmentation cardiaque (M1) et validation externe.

Détails, rôle de chaque jeu et structure recommandée : [`datasets/README.md`](./datasets/README.md).

---

## Reproduire les résultats

```bash
pip install torch torchvision opencv-python pandas numpy matplotlib seaborn \
            scikit-learn scipy pillow pydicom
```

1. Télécharger les trois jeux de données publics (voir `datasets/README.md`).
2. Entraîner ou récupérer les checkpoints des modules M1–M4 (disponibles dans [`Etudes/Models/`](./Etudes/Models)).
3. Exécuter [`Interferences/interference.ipynb`](./Interferences/interference.ipynb) pour générer le CSV de résultats sur un jeu d'images.
4. Exécuter les notebooks du dossier [`Etudes/`](./Etudes) pour reproduire les six études de validation.
5. Utiliser les notebooks de [`pipline/`](./pipline) pour tester le pipeline sur une image individuelle.

Chaque sous-dossier a son propre README avec les chemins exacts à adapter (les notebooks ont été écrits pour Kaggle) et le détail de ses notebooks.

---

## Limites méthodologiques (documentées dans le rapport)

- La vérité terrain utilisée pour l'évaluation (étiquettes NIH) est extraite par traitement du langage naturel de comptes-rendus radiologiques, et non par confirmation clinique d'événements cardiovasculaires.
- Le diamètre thoracique n'est pas segmenté mais approximé à 75 % de la largeur de l'image.
- Les pondérations de la formule V5 ont été optimisées sur le même échantillon de 501 images que celui utilisé pour l'évaluation (surajustement estimé et quantifié dans l'Étude 6).
- M2 (segmentation calcique) est conservé pour la visualisation mais n'entre pas dans le score final.

Ce projet est un travail de recherche académique et **ne constitue pas un dispositif de diagnostic clinique**.
