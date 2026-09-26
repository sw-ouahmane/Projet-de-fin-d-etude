# Pipeline V3 — Application Locale Windows
## Guide d'installation et de lancement

---

## Structure du dossier

```
ton_dossier/
├── app_pipeline_v3.py       ← l'application
├── README_INSTALLATION.md   ← ce fichier
└── models/
    ├── unet_cardio_robust_best.pth
    ├── unet_calcif_best_kagg.pth
    ├── calcif_model_best.pth
    └── cardio_risk_v2_best.pth
```

---

## Étape 1 — Installer les dépendances (une seule fois)

Ouvre **cmd** ou **PowerShell** et tape :

```bash
pip install streamlit torch torchvision pillow opencv-python-headless numpy
```

Si tu as un GPU NVIDIA, remplace `torch torchvision` par :
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Étape 2 — Placer tes modèles

Place tes 4 fichiers `.pth` dans un dossier `models/` :

```
models/
├── unet_cardio_robust_best.pth    ← M1 U-Net Cardio
├── unet_calcif_best_kagg.pth      ← M2 U-Net Calcif
├── calcif_model_best.pth          ← M3 CNN Calcif
└── cardio_risk_v2_best.pth        ← M4 CNN Risque
```

> **Note :** si tes fichiers ont des noms légèrement différents
> (ex: `unet_calcif_best_kagg (2).pth`), renomme-les sans les espaces et parenthèses.

---

## Étape 3 — Lancer l'application

Dans **cmd** ou **PowerShell**, navigue vers ton dossier :

```bash
cd C:\Users\Abdallah\ton_dossier
streamlit run app_pipeline_v3.py
```

L'application s'ouvre automatiquement dans le navigateur à l'adresse :
**http://localhost:8501**

---

## Étape 4 — Utiliser l'application

1. Dans la **sidebar gauche**, mets le chemin vers ton dossier `models/`
   - Exemple : `C:\Users\Abdallah\models`
2. Vérifie que les 4 modèles sont trouvés (**✅ vert**)
3. Dans la zone principale, **charge une radiographie** (PNG, JPG, BMP)
4. Clique **"▶ Lancer l'analyse Pipeline V3"**
5. Attends quelques secondes (CPU local) → résultats affichés

---

## Temps d'analyse estimé (CPU)

| Phase | Temps |
|---|---|
| Chargement modèles (1ère fois) | ~30 secondes |
| Chargement modèles (cache) | ~2 secondes |
| Analyse une image | ~15-30 secondes |

> Les modèles sont mis en cache — la 2ème analyse est beaucoup plus rapide.

---

## Pour la soutenance

**Image recommandée pour la démo :**
- Télécharge `00000001_000.png` depuis le dataset NIH (Cardiomegaly confirmé)
- Résultat attendu : CTR=0.651, Score=45.0%, Niveau ÉLEVÉ

**Ordre de présentation suggéré :**
1. Montrer l'interface vide → expliquer le pipeline V3
2. Charger l'image NIH Cardiomegaly → lancer → montrer CTR=0.651
3. Charger une image Normale → montrer que le score baisse
4. Expliquer la formule 0.45×CTR + 0.55×Cong affichée

---

## En cas de problème

**Erreur "Module not found"**
```bash
pip install streamlit torch torchvision pillow opencv-python-headless numpy
```

**Erreur "FileNotFoundError"**
→ Vérifie le chemin des modèles dans la sidebar

**Erreur "RuntimeError: size mismatch"**
→ Assure-toi que chaque .pth correspond au bon modèle

**Port déjà utilisé**
```bash
streamlit run app_pipeline_v3.py --server.port 8502


