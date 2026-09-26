"""
APPLICATION STREAMLIT — PIPELINE V5
=====================================
Interface locale pour le dépistage cardiovasculaire opportuniste
Formule finale V5 (4 composantes) — cohérente avec le rapport de PFE

  risk_global = (0.15×s_ctr + 0.40×s_cong + 0.10×s_calcif + 0.35×prob_cardio) × 100

  AUC = 0.8916  IC95% [0.8572 ; 0.9229]  —  n = 501 images NIH ChestX-ray14
  Seuil ÉLEVÉ ≥ 35 %  ·  MODÉRÉ ≥ 25 %  ·  INDÉTERMINÉ si CTR ∉ [0.30 ; 0.75]

INSTALLATION (une seule fois) :
  pip install streamlit torch torchvision pillow opencv-python-headless numpy

LANCEMENT :
  streamlit run app_pipeline_v5.py

STRUCTURE DOSSIER :
  ton_dossier/
  ├── app_pipeline_v5.py          ← ce fichier
  └── models/
      ├── unet_cardio_robust_best.pth      (M1)
      ├── unet_calcif_best_kagg.pth        (M2)
      ├── calcif_model_final.pth           (M3 — checkpoint RÉELLEMENT entraîné)
      └── cardio_risk_v2_best.pth          (M4)

ATTENTION : le checkpoint M3 valide est bien `calcif_model_final.pth` (epoch 20).
`calcif_model_best.pth` est un EfficientNet ImageNet standard, PAS le modèle entraîné.
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import glob
from torchvision import transforms, models
from PIL import Image
import os

# ============================================================
# CONFIGURATION PAGE
# ============================================================
st.set_page_config(
    page_title="Pipeline V5 — Risque Cardiovasculaire",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .main { background: #F0F4F8; }
  .stButton > button {
    background: #1D9E75; color: white; font-weight: 700; border: none;
    border-radius: 8px; padding: 12px 24px; font-size: 16px; width: 100%;
  }
  .stButton > button:hover { background: #15866A; }
  .score-eleve   { color: #DC2626; font-size: 48px; font-weight: 800; }
  .score-modere  { color: #F59E0B; font-size: 48px; font-weight: 800; }
  .score-faible  { color: #16A34A; font-size: 48px; font-weight: 800; }
  .score-indet   { color: #7D3C98; font-size: 48px; font-weight: 800; }
  .formula-box {
    background: #1B2A4A; color: #34D399; border-radius: 10px; padding: 14px;
    font-family: monospace; font-size: 14px; font-weight: 700;
    text-align: center; margin: 12px 0;
  }
  .module-header {
    background: #1B2A4A; color: white; padding: 8px 14px;
    border-radius: 8px 8px 0 0; font-weight: 700; font-size: 13px;
  }
  .note-exclu {
    background: #F3F4F6; border-left: 3px solid #9CA3AF; padding: 8px 12px;
    font-size: 11px; color: #6B7280; border-radius: 0 4px 4px 0;
  }
  .note-indet {
    background: #F5EEF8; border-left: 3px solid #7D3C98; padding: 10px 14px;
    font-size: 12px; color: #4A235A; border-radius: 0 4px 4px 0;
  }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ARCHITECTURES MODÈLES
# ============================================================
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
            )
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = double_conv(512, 1024)
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = double_conv(1024, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = double_conv(512, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = double_conv(256, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = double_conv(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([e4, self.up4(b)],  dim=1))
        d3 = self.dec3(torch.cat([e3, self.up3(d4)], dim=1))
        d2 = self.dec2(torch.cat([e2, self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([e1, self.up1(d2)], dim=1))
        return torch.sigmoid(self.final(d1))


class EfficientNetCNN(nn.Module):
    """Architecture partagée par M3 et M4 : EfficientNet-B0 monocanal, 3 classes."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        orig = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, orig.out_channels,
            kernel_size=orig.kernel_size, stride=orig.stride,
            padding=orig.padding, bias=False
        )
        in_f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 256),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# CONSTANTES — PIPELINE V5
# ============================================================
CAT_RISK   = ['Normal', 'Cardiomegaly', 'Edema']
CAT_CALCIF = ['Normal', 'Calcification', 'Aortic_enlargement']

# Poids de fusion V5 (somme = 1.00)
W_CTR    = 0.15   # M1 — cardiomégalie morphométrique
W_CONG   = 0.40   # M4 — probabilité Edema (congestion)
W_CALCIF = 0.10   # M3 — score calcification / élargissement aortique
W_CARDIO = 0.35   # M4 — probabilité Cardiomegaly (corrige l'anti-corrélation V1/V3)

SEUIL_ELEVE  = 35     # ≥ 35 %  → ÉLEVÉ
SEUIL_MODERE = 25     # ≥ 25 %  → MODÉRÉ
SEUIL_CONFIANCE_M3 = 0.60   # sous ce seuil, s_calcif est neutralisé à 0
CTR_MIN, CTR_MAX   = 0.30, 0.75   # plausibilité anatomique → INDÉTERMINÉ hors bornes

# Performances validées (n = 501 NIH, cf. rapport §Résultats)
AUC_V5      = 0.8916
AUC_IC      = (0.8572, 0.9229)
SENSIBILITE = 0.928
SPECIFICITE = 0.766
VPP         = 0.888

# Fichiers modèles attendus (motifs glob — tolère espaces / suffixes)
PATTERNS = {
    'M1': 'unet_cardio_robust*best*.pth',
    'M2': 'unet_calcif*best*.pth',
    'M3': 'calcif_model_final*.pth',
    'M4': 'cardio_risk*v2*best*.pth',
}

T512 = transforms.Compose([
    transforms.Resize((512, 512)), transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
T224 = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def trouver(models_dir, pattern):
    """Résout un chemin modèle via glob (les noms de fichiers contiennent
    parfois des espaces ou des suffixes de téléchargement)."""
    hits = sorted(glob.glob(os.path.join(models_dir, pattern)))
    return hits[0] if hits else None


# ============================================================
# CHARGEMENT MODÈLES — VALIDATION STRICTE
# ============================================================
@st.cache_resource
def charger_modeles(models_dir):
    """Chargement STRICT de tous les modules.

    Le mode strict=False a masqué pendant plusieurs cycles un checkpoint M3
    invalide (313 clés manquantes). M3 contribuant désormais 10 % du score,
    tout échec de chargement doit lever une exception explicite.
    """
    device = torch.device('cpu')
    rapport = {}

    m1 = UNet().to(device)
    m1.load_state_dict(torch.load(trouver(models_dir, PATTERNS['M1']),
                                  map_location=device, weights_only=False))
    m1.eval(); rapport['M1'] = 'strict OK'

    m2 = UNet().to(device)
    m2.load_state_dict(torch.load(trouver(models_dir, PATTERNS['M2']),
                                  map_location=device, weights_only=False))
    m2.eval(); rapport['M2'] = 'strict OK'

    # M3 — EfficientNet-B0 monocanal 3 classes (calcif_model_final.pth, epoch 20)
    m3 = EfficientNetCNN(num_classes=3).to(device)
    ck3 = torch.load(trouver(models_dir, PATTERNS['M3']),
                     map_location=device, weights_only=False)
    if isinstance(ck3, dict) and 'state_dict' in ck3:
        ck3 = ck3['state_dict']
    res3 = m3.load_state_dict(ck3, strict=False)
    if res3.missing_keys:
        raise RuntimeError(
            f"M3 : {len(res3.missing_keys)} clés manquantes — checkpoint invalide. "
            f"Vérifie que le fichier est bien calcif_model_final.pth (epoch 20) "
            f"et non calcif_model_best.pth (EfficientNet ImageNet standard)."
        )
    m3.eval(); rapport['M3'] = 'strict OK (0 clé manquante)'

    m4 = EfficientNetCNN(num_classes=3).to(device)
    ck4 = torch.load(trouver(models_dir, PATTERNS['M4']),
                     map_location=device, weights_only=False)
    if isinstance(ck4, dict) and 'state_dict' in ck4:
        ck4 = ck4['state_dict']
    m4.load_state_dict(ck4)
    m4.eval(); rapport['M4'] = 'strict OK'

    return m1, m2, m3, m4, device, rapport


# ============================================================
# CALCUL DU CTR
# ============================================================
def calculer_ctr(mask, position='PA'):
    """Le CTR est un rapport de deux longueurs : il est invariant par
    changement d'échelle isotrope, aucun spacing n'est donc nécessaire.
    Le diamètre thoracique est approximé à 75 % de la largeur (limite §7.4.1).
    """
    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heart_px = 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, wb, hb = cv2.boundingRect(c)
        cx, cy = x + wb // 2, y + hb // 2
        if w * 0.20 < cx < w * 0.80 and h * 0.20 < cy < h * 0.80:
            heart_px = wb
    thorax_px = int(w * 0.75)
    ctr = heart_px / thorax_px if (thorax_px > 0 and heart_px > 0) else 0.0
    # Correction de la magnification en incidence antéro-postérieure
    if 'AP' in str(position).upper():
        ctr = max(ctr - 0.05, 0.0)
    ctr_valide = (CTR_MIN <= ctr <= CTR_MAX)
    return ctr, heart_px, thorax_px, ctr_valide


# ============================================================
# PIPELINE V5 — ANALYSE
# ============================================================
def analyser(img_pil, m1, m2, m3, m4, device, position='PA'):
    img_np = np.array(img_pil.resize((512, 512)))
    inp512 = T512(img_pil).unsqueeze(0).to(device)
    inp224 = T224(img_pil).unsqueeze(0).to(device)

    # --- M1 : segmentation cardiaque → CTR ---
    with torch.no_grad():
        mask_c = (m1(inp512).cpu().squeeze().numpy() > 0.7).astype(np.uint8)
    ctr, hpx, tpx, ctr_valide = calculer_ctr(mask_c, position)
    s_ctr = min(max((ctr - 0.40) / 0.20, 0.0), 1.0)

    # --- M2 : surface calcifiée (visualisation, hors fusion : +0.0002) ---
    with torch.no_grad():
        mask_k = (m2(inp512).cpu().squeeze().numpy() > 0.5).astype(np.uint8)
    calcif_pct = float(mask_k.sum()) / mask_k.size * 100

    # --- M3 : classification calcifications (10 % du score) ---
    with torch.no_grad():
        prob3 = torch.softmax(m3(inp224), dim=1).cpu().numpy()[0]
    calcif_label = CAT_CALCIF[int(np.argmax(prob3))]
    conf_m3      = float(prob3.max())
    m3_utilise   = conf_m3 >= SEUIL_CONFIANCE_M3
    s_calcif     = min(float(prob3[1]) + float(prob3[2]), 1.0) if m3_utilise else 0.0

    # --- M4 : classification du risque (75 % du score : congestion + cardiomégalie) ---
    with torch.no_grad():
        prob4 = torch.softmax(m4(inp224), dim=1).cpu().numpy()[0]
    risk_label  = CAT_RISK[int(np.argmax(prob4))]
    prob_normal = float(prob4[0])
    prob_cardio = float(prob4[1])
    s_cong      = float(prob4[2])

    # --- Fusion V5 ---
    score = round((W_CTR * s_ctr + W_CONG * s_cong
                   + W_CALCIF * s_calcif + W_CARDIO * prob_cardio) * 100, 1)

    if not ctr_valide:
        niveau = 'INDÉTERMINÉ'
    elif score >= SEUIL_ELEVE:
        niveau = 'ÉLEVÉ'
    elif score >= SEUIL_MODERE:
        niveau = 'MODÉRÉ'
    else:
        niveau = 'FAIBLE'

    # --- Overlay masque cardiaque ---
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    overlay = img_rgb.copy()
    overlay[cv2.resize(mask_c, (512, 512)) > 0] = [200, 50, 50]
    img_overlay = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)

    return {
        'ctr': round(ctr, 3), 's_ctr': round(s_ctr, 3),
        'ctr_valide': ctr_valide, 'cardio_flag': ctr > 0.50,
        'heart_px': hpx, 'thorax_px': tpx,
        'calcif_pct': round(calcif_pct, 2),
        'calcif_label': calcif_label, 'conf_m3': round(conf_m3, 3),
        'm3_utilise': m3_utilise, 's_calcif': round(s_calcif, 3),
        'prob_normal': round(prob_normal, 3),
        'prob_cardio': round(prob_cardio, 3),
        'prob_edema':  round(s_cong, 3),
        's_cong': round(s_cong, 3),
        'risk_label': risk_label,
        'score': score, 'niveau': niveau,
        'mask_overlay': img_overlay, 'img_np': img_np,
    }


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🫀 Pipeline V5")
    st.markdown("**Risque Cardiovasculaire par Radio Thoracique**")
    st.markdown("---")

    st.markdown("### 📁 Dossier des modèles")
    models_dir = st.text_input(
        "Chemin vers tes modèles .pth",
        value=r"C:\Users\dell\Desktop\PFE\App\models",
        help="Dossier contenant les 4 fichiers .pth"
    )

    trouves = {k: trouver(models_dir, p) for k, p in PATTERNS.items()}
    all_ok  = all(v is not None for v in trouves.values())

    if all_ok:
        st.success("✅ Les 4 modèles sont trouvés")
        with st.expander("Fichiers résolus"):
            for k, v in trouves.items():
                st.markdown(f"`{k} → {os.path.basename(v)}`")
    else:
        st.error("❌ Modèles manquants — vérifie le chemin")
        for k, p in PATTERNS.items():
            icon = "✅" if trouves[k] else "❌"
            st.markdown(f"`{icon} {k} : {p}`")

    st.markdown("---")
    st.markdown("### 🩻 Incidence")
    position = st.radio(
        "Position de la prise de vue",
        ['PA', 'AP'], horizontal=True,
        help="En AP, le cœur est magnifié : correction de −0.05 sur le CTR"
    )

    st.markdown("---")
    st.markdown("### 📋 Formule V5")
    st.code(
        "risk = (0.15×s_ctr + 0.40×s_cong\n"
        "      + 0.10×s_calcif + 0.35×p_cardio)×100",
        language=None
    )
    st.markdown(
        f"**AUC = {AUC_V5:.4f}**  IC95 % [{AUC_IC[0]:.4f} ; {AUC_IC[1]:.4f}]"
    )
    st.markdown(f"Seuil ÉLEVÉ ≥ {SEUIL_ELEVE} % · MODÉRÉ ≥ {SEUIL_MODERE} %")
    st.markdown(f"INDÉTERMINÉ si CTR ∉ [{CTR_MIN} ; {CTR_MAX}]")

    st.markdown("---")
    st.markdown("### 📊 Justification des poids")
    st.markdown(f"""
- **Se = {SENSIBILITE:.3f} · Sp = {SPECIFICITE:.3f} · VPP = {VPP:.3f}** au seuil {SEUIL_ELEVE} %
- `prob_cardio` corrige l'anti-corrélation du sous-groupe Cardiomegaly (AUC 0.3523 en V1/V3)
- **M3** : +0.0068 AUC (positif dans 81 % des 200 partitions)
- **M2** : +0.0002 → exclu de la fusion (Étude 2B)
- Surapprentissage estimé : **+0.0007** (200 bootstraps)
    """)

# ============================================================
# CONTENU PRINCIPAL
# ============================================================
st.markdown("# 🫀 Dépistage du Risque Cardiovasculaire")
st.markdown("**Pipeline V5 · Trustworthy AI · Master MIT (TAM) · Abdallah Ouahmane · 2025/2026**")
st.markdown("---")

st.markdown("### 📤 Charger une radiographie thoracique")
uploaded = st.file_uploader(
    "Formats acceptés : PNG, JPG, JPEG, BMP",
    type=['png', 'jpg', 'jpeg', 'bmp'],
    help="Radiographie thoracique en niveaux de gris"
)

if uploaded is not None:
    img_pil = Image.open(uploaded).convert('L')
    col_img, col_info = st.columns([1, 2])

    with col_img:
        st.image(img_pil, caption="Radio chargée", use_container_width=True)
        st.markdown(f"**Taille :** {img_pil.size[0]}×{img_pil.size[1]} px")

    with col_info:
        st.markdown("### ℹ️ Image chargée avec succès")
        st.markdown(f"**Fichier :** `{uploaded.name}` · **Incidence :** {position}")
        st.markdown("""
**Ce que le pipeline va faire :**
1. 🔬 Prétraitement 512×512 / 224×224
2. 🫀 M1 — Segmentation cardiaque → CTR → **s_ctr (15 %)**
3. 🔬 M2 — Segmentation calcifications (visualisation seule)
4. 🤖 M3 — Classification calcifications → **s_calcif (10 %)**
5. 🫁 M4 — Classification risque → **s_cong (40 %)** + **prob_cardio (35 %)**
6. ⚡ Fusion V5 puis contrôle de plausibilité du CTR
        """)

    st.markdown("---")

    if all_ok:
        if st.button("▶ Lancer l'analyse Pipeline V5", type="primary"):
            try:
                with st.spinner("Chargement strict des 4 modules..."):
                    m1, m2, m3, m4, device, rapport = charger_modeles(models_dir)
                st.success("✅ Chargement : " + " · ".join(
                    f"{k} {v}" for k, v in rapport.items()))
            except Exception as e:
                st.error(f"❌ Échec du chargement des modèles\n\n{e}")
                st.stop()

            progress = st.progress(0, text="M1 U-Net cardiaque...")
            r = analyser(img_pil, m1, m2, m3, m4, device, position)
            progress.progress(100, text="Analyse terminée")
            st.success("✅ Analyse terminée")

            # ── RÉSULTATS ────────────────────────────────────────
            st.markdown("---")
            st.markdown("## 📊 Résultats — Pipeline V5")

            if not r['ctr_valide']:
                st.markdown(
                    f'<div class="note-indet">'
                    f'⚠ <b>Cas INDÉTERMINÉ</b> — le CTR mesuré ({r["ctr"]:.3f}) sort '
                    f'des bornes de plausibilité anatomique [{CTR_MIN} ; {CTR_MAX}]. '
                    f'La segmentation cardiaque n\'est pas exploitable sur cette image : '
                    f'le score calculé ({r["score"]} %) est affiché à titre indicatif '
                    f'mais ne doit pas être interprété cliniquement. '
                    f'Relecture radiologique requise.'
                    f'</div>', unsafe_allow_html=True
                )

            col1, col2, col3 = st.columns(3)

            with col1:
                couleur = {'ÉLEVÉ': "🔴", 'MODÉRÉ': "🟡",
                           'FAIBLE': "🟢", 'INDÉTERMINÉ': "🟣"}[r['niveau']]
                css = {'ÉLEVÉ': "score-eleve", 'MODÉRÉ': "score-modere",
                       'FAIBLE': "score-faible", 'INDÉTERMINÉ': "score-indet"}[r['niveau']]
                st.metric(label=f"{couleur} Risque Cardiovasculaire",
                          value=f"{r['score']}%", delta=f"Niveau : {r['niveau']}")
                st.markdown(f'<div class="{css}">{r["score"]}%</div>',
                            unsafe_allow_html=True)
                st.markdown(f"**[{r['niveau']}]**")

            with col2:
                st.metric("CTR (M1)", f"{r['ctr']:.3f}",
                          delta="⚠ Cardiomégalie" if r['cardio_flag'] else "✓ Normal",
                          delta_color="inverse" if r['cardio_flag'] else "normal")
                st.metric("s_ctr → 15 %",
                          f"{W_CTR * r['s_ctr'] * 100:.1f} / 15 pts",
                          delta=f"s_ctr = {r['s_ctr']:.3f}")
                st.metric("s_calcif → 10 %",
                          f"{W_CALCIF * r['s_calcif'] * 100:.1f} / 10 pts",
                          delta=f"s_calcif = {r['s_calcif']:.3f}")

            with col3:
                st.metric("s_cong (Edema) → 40 %",
                          f"{W_CONG * r['s_cong'] * 100:.1f} / 40 pts",
                          delta=f"s_cong = {r['s_cong']:.3f}")
                st.metric("prob_cardio → 35 %",
                          f"{W_CARDIO * r['prob_cardio'] * 100:.1f} / 35 pts",
                          delta=f"p_cardio = {r['prob_cardio']:.3f}")
                st.metric("Diagnostic M4", r['risk_label'])

            st.markdown(
                f'<div class="formula-box">'
                f'risk_global = (0.15×{r["s_ctr"]:.3f} + 0.40×{r["s_cong"]:.3f} '
                f'+ 0.10×{r["s_calcif"]:.3f} + 0.35×{r["prob_cardio"]:.3f}) × 100 '
                f'= {r["score"]} %'
                f'</div>', unsafe_allow_html=True
            )

            # Décomposition des contributions
            st.markdown("#### Décomposition du score")
            contrib = {
                'M1 — s_ctr (15 %)':        W_CTR    * r['s_ctr']       * 100,
                'M4 — s_cong (40 %)':       W_CONG   * r['s_cong']      * 100,
                'M3 — s_calcif (10 %)':     W_CALCIF * r['s_calcif']    * 100,
                'M4 — prob_cardio (35 %)':  W_CARDIO * r['prob_cardio'] * 100,
            }
            for nom, val in contrib.items():
                st.progress(min(val / 40, 1.0), text=f"{nom} : {val:.1f} pts")

            # ── MODULES DÉTAILLÉS ────────────────────────────────
            st.markdown("---")
            st.markdown("### 🔍 Détail par module")

            col_m1, col_m4 = st.columns(2)

            with col_m1:
                st.markdown('<div class="module-header">🫀 M1 — U-Net Cardiaque '
                            '(Dice 0.9580 · contribue 15 %)</div>',
                            unsafe_allow_html=True)
                st.markdown(f"""
| Paramètre | Valeur |
|---|---|
| CTR mesuré ({position}) | **{r['ctr']:.3f}** |
| Plausibilité [{CTR_MIN} ; {CTR_MAX}] | **{'✓ valide' if r['ctr_valide'] else '⚠ hors bornes'}** |
| Cardiomégalie (CTR > 0.50) | **{'⚠ OUI' if r['cardio_flag'] else '✓ NON'}** |
| s_ctr normalisé | **{r['s_ctr']:.3f}** |
| Contribution | **{W_CTR * r['s_ctr'] * 100:.1f} / 15 pts** |
                """)
                st.image(r['mask_overlay'], caption="Masque cardiaque U-Net",
                         use_container_width=True)

            with col_m4:
                st.markdown('<div class="module-header">🫁 M4 — CNN Risque EfficientNet-B0 '
                            '(AUC 0.8954 · contribue 75 %)</div>',
                            unsafe_allow_html=True)
                st.markdown(f"""
| Classe | Probabilité | Rôle dans V5 |
|---|---|---|
| Normal | **{r['prob_normal']*100:.1f}%** | — |
| Cardiomegaly | **{r['prob_cardio']*100:.1f}%** | → **prob_cardio (35 %)** |
| Edema | **{r['prob_edema']*100:.1f}%** | → **s_cong (40 %)** |
| Diagnostic | **{r['risk_label']}** | |
| Contribution totale | **{(W_CONG*r['s_cong'] + W_CARDIO*r['prob_cardio'])*100:.1f} / 75 pts** | |
                """)
                st.progress(r['prob_normal'], text=f"Normal : {r['prob_normal']*100:.1f}%")
                st.progress(r['prob_cardio'], text=f"Cardiomegaly (→ 35 %) : {r['prob_cardio']*100:.1f}%")
                st.progress(r['prob_edema'],  text=f"Edema (→ 40 %) : {r['prob_edema']*100:.1f}%")

            # M3 + M2
            st.markdown("---")
            col_m3, col_m2 = st.columns(2)

            with col_m3:
                st.markdown('<div class="module-header">🤖 M3 — CNN Calcifications '
                            '(macro F1 0.688 · contribue 10 %)</div>',
                            unsafe_allow_html=True)
                st.markdown(f"""
| Paramètre | Valeur |
|---|---|
| Classe prédite | **{r['calcif_label']}** |
| Confiance max | **{r['conf_m3']*100:.1f}%** |
| Seuil de confiance | {SEUIL_CONFIANCE_M3*100:.0f}% |
| Statut | **{'✓ retenu' if r['m3_utilise'] else '⚠ neutralisé (s_calcif = 0)'}** |
| s_calcif = p(Calcif) + p(Aortic) | **{r['s_calcif']:.3f}** |
| Contribution | **{W_CALCIF * r['s_calcif'] * 100:.1f} / 10 pts** |
                """)
                if not r['m3_utilise']:
                    st.markdown(
                        '<div class="note-exclu">Confiance inférieure au seuil de '
                        f'{SEUIL_CONFIANCE_M3:.2f} : s_calcif est neutralisé à 0 pour '
                        'éviter d\'injecter une prédiction non fiable dans la fusion.'
                        '</div>', unsafe_allow_html=True)

            with col_m2:
                st.markdown('<div class="module-header">🔬 M2 — U-Net Calcifications '
                            '(Dice 0.793 · hors fusion)</div>',
                            unsafe_allow_html=True)
                st.metric("Surface calcifiée détectée", f"{r['calcif_pct']:.2f}%")
                st.markdown(
                    '<div class="note-exclu">'
                    'ℹ Module exclu de la fusion : gain mesuré de <b>+0.0002 AUC</b> '
                    '(Étude 2B), négligeable au regard du coût de calcul. '
                    'Affiché à titre d\'information clinique complémentaire.'
                    '</div>', unsafe_allow_html=True)

            # ── RÉSUMÉ FINAL ─────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📋 Résumé Final")

            color = {'ÉLEVÉ': '#DC2626', 'MODÉRÉ': '#F59E0B',
                     'FAIBLE': '#16A34A', 'INDÉTERMINÉ': '#7D3C98'}[r['niveau']]

            st.markdown(f"""
<div style="background:white; border-radius:12px; padding:20px; border: 2px solid {color}">
  <h3 style="color:{color}">Score : {r['score']}% — Niveau {r['niveau']}</h3>
  <p style="font-size:13px; color:#64748B; margin-top:8px">
    CTR = {r['ctr']:.3f} {'⚠ Cardiomégalie' if r['cardio_flag'] else '✓ Normal'} |
    Congestion = {r['prob_edema']*100:.1f}% |
    Cardiomégalie (M4) = {r['prob_cardio']*100:.1f}% |
    Calcification = {r['s_calcif']:.3f} |
    Diagnostic M4 = {r['risk_label']}
  </p>
  <p style="font-family:monospace; font-size:12px; background:#1B2A4A; color:#34D399;
            border-radius:8px; padding:10px; margin-top:10px; text-align:center">
    risk_global = (0.15×{r['s_ctr']:.3f} + 0.40×{r['s_cong']:.3f}
    + 0.10×{r['s_calcif']:.3f} + 0.35×{r['prob_cardio']:.3f}) × 100 = {r['score']}%
  </p>
  <p style="font-size:11px; color:#64748B; margin-top:8px">
    ★ AUC = {AUC_V5:.4f} · IC95 % [{AUC_IC[0]:.4f} ; {AUC_IC[1]:.4f}] ·
    Se = {SENSIBILITE:.3f} · Sp = {SPECIFICITE:.3f} · VPP = {VPP:.3f} au seuil {SEUIL_ELEVE} % ·
    n = 501 images NIH ChestX-ray14
  </p>
  <p style="font-size:10px; color:#9CA3AF; margin-top:4px">
    ⚠ Outil de dépistage opportuniste uniquement — ne constitue pas un diagnostic
    médical — toute décision clinique requiert la confirmation d'un radiologue.
  </p>
</div>
            """, unsafe_allow_html=True)

    else:
        st.error("❌ Configure le chemin des modèles dans la sidebar avant de lancer l'analyse.")

else:
    st.markdown("### 👆 Charge une radiographie thoracique pour commencer")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**📌 Format attendu**
- Radiographie thoracique PA ou AP
- Niveaux de gris (L)
- PNG, JPG, JPEG, BMP
- Toute résolution acceptée
        """)
    with col2:
        st.markdown("""
**⚙️ Pipeline V5**
- M1 U-Net → s_ctr (15 %)
- M4 CNN → s_cong (40 %)
- M3 CNN → s_calcif (10 %)
- M4 CNN → prob_cardio (35 %)
- M2 → visualisation seule
        """)
    with col3:
        st.markdown(f"""
**📊 Performance validée**
- AUC = {AUC_V5:.4f} [{AUC_IC[0]:.4f} ; {AUC_IC[1]:.4f}]
- Se = {SENSIBILITE:.3f} · Sp = {SPECIFICITE:.3f}
- Surapprentissage : +0.0007
- 501 images NIH · 200 bootstraps
        """)

    st.info("💡 **Conseil soutenance :** charge l'image NIH 00000001_000.png "
            "(Cardiomegaly confirmé) — c'est le cas qui met en évidence l'apport "
            "de `prob_cardio`, absent des formules V1/V3.")