# Chest X-Ray Datasets

This repository uses three public chest X-ray datasets for the development, evaluation, and analysis of deep learning models for thoracic abnormality detection, classification, localization, and image enhancement.

## Datasets

1. [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)
2. [NIH ChestX-ray14 – CLAHE Enhanced](https://www.kaggle.com/datasets/rahulogoel/clahe-enhancement-on-chestx-ray14)
3. [JSRT – Japanese Society of Radiological Technology Database](http://db.jsrt.or.jp/eng.php)

> **Note:** The datasets themselves are not stored in this GitHub repository because of their large size and distribution conditions. They must be downloaded separately from their respective sources.

---

## 1. VinBigData Chest X-ray Abnormalities Detection

### Description

The **VinBigData Chest X-ray Abnormalities Detection** dataset is a large-scale chest radiography dataset designed for the detection and localization of thoracic abnormalities.

The dataset contains **18,000 chest X-ray scans**:

* 15,000 training images
* 3,000 test images
* DICOM format
* Images acquired in PA view
* Radiologist annotations
* Bounding-box annotations for abnormalities

### Main Tasks

The dataset can be used for:

* Thoracic abnormality classification
* Object detection
* Abnormality localization
* Multi-class classification
* Multi-label analysis

### Abnormality Classes

| ID | Abnormality        |
| -: | ------------------ |
|  0 | Aortic enlargement |
|  1 | Atelectasis        |
|  2 | Calcification      |
|  3 | Cardiomegaly       |
|  4 | Consolidation      |
|  5 | ILD                |
|  6 | Infiltration       |
|  7 | Lung Opacity       |
|  8 | Nodule/Mass        |
|  9 | Other lesion       |
| 10 | Pleural effusion   |
| 11 | Pleural thickening |
| 12 | Pneumothorax       |
| 13 | Pulmonary fibrosis |
| 14 | No finding         |

### Annotations

The training annotations include information such as:

* `image_id`
* `class_name`
* `class_id`
* `rad_id`
* `x_min`
* `y_min`
* `x_max`
* `y_max`

The bounding boxes identify the location of the detected abnormalities.

### Source

[Kaggle – VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)

---

# 2. NIH ChestX-ray14 – CLAHE Enhanced

## Description

**ChestX-ray14** is a large-scale chest X-ray dataset released by the **National Institutes of Health (NIH)**.

The original ChestX-ray14 dataset contains approximately **112,120 frontal-view chest X-ray images from 30,805 patients**, covering 14 thoracic pathologies.

The version used in this project is a **CLAHE-enhanced version** of ChestX-ray14.

### CLAHE

**CLAHE** stands for:

> **Contrast Limited Adaptive Histogram Equalization**

It is an image enhancement technique designed to improve local contrast while limiting excessive amplification of image noise.

CLAHE can make anatomical structures and pathological regions more visible.

### Main Tasks

This dataset can be used for:

* Multi-label classification
* Image enhancement
* Comparison between original and enhanced images
* Deep learning classification
* Robustness analysis
* Contrast enhancement experiments

### Pathologies

The original ChestX-ray14 dataset contains 14 pathology labels:

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

A `No Finding` category is also commonly used for images without any of the listed findings.

### Purpose in the Project

The CLAHE-enhanced dataset can be used to investigate whether image enhancement improves model performance.

Possible comparisons include:

```text
Original X-ray
      ↓
Classification Model
      ↓
Performance

CLAHE-enhanced X-ray
      ↓
Classification Model
      ↓
Performance
```

Performance can be evaluated using:

* Accuracy
* AUC
* Macro AUC
* F1-score
* Sensitivity
* Specificity

### Source

[Kaggle – CLAHE Enhancement on ChestX-ray14](https://www.kaggle.com/datasets/rahulogoel/clahe-enhancement-on-chestx-ray14)

---

# 3. JSRT – Japanese Society of Radiological Technology Database

## Description

The **Japanese Society of Radiological Technology (JSRT)** database is a well-known chest radiograph dataset used in medical image analysis and computer-aided diagnosis research.

The dataset contains **247 chest radiographs**:

* 154 images with pulmonary nodules
* 93 images without pulmonary nodules

The images are high-resolution chest radiographs and are commonly used for pulmonary nodule detection and segmentation research.

### Main Tasks

The JSRT dataset can be used for:

* Pulmonary nodule detection
* Lung segmentation
* Medical image analysis
* Computer-aided diagnosis
* Image processing
* External validation

### Purpose in the Project

Because JSRT comes from a different dataset and acquisition environment, it can be useful as an **external validation dataset**.

For example:

```text
Training Dataset
       ↓
Deep Learning Model
       ↓
Internal Evaluation
       ↓
JSRT Dataset
       ↓
External Validation
```

This allows us to investigate whether a model trained on another dataset can generalize to a different population and imaging source.

### Source

[JSRT Database – Japanese Society of Radiological Technology](http://db.jsrt.or.jp/eng.php)

---

# Dataset Comparison

| Dataset                | Number of Images | Main Task                       | Annotation         | Main Purpose                       |
| ---------------------- | ---------------: | ------------------------------- | ------------------ | ---------------------------------- |
| **VinBigData**         |           18,000 | Detection / Classification      | Bounding Boxes     | Abnormality localization           |
| **ChestX-ray14 CLAHE** |         112,120* | Classification                  | Image-level labels | Image enhancement & classification |
| **JSRT**               |              247 | Nodule Detection / Segmentation | Nodule annotations | External validation                |

* The 112,120 figure refers to the original ChestX-ray14 dataset. The exact number of images in the CLAHE-enhanced Kaggle version should be verified after downloading the dataset.

---

# Recommended Repository Structure

The raw datasets should **not** be uploaded to GitHub because of their large size.

Recommended project structure:

```text
Projet-de-fin-d-etude/
│
├── data/
│   ├── vinbigdata/
│   │   ├── train/
│   │   ├── test/
│   │   └── train.csv
│   │
│   ├── chestxray14_clahe/
│   │   ├── images/
│   │   └── labels/
│   │
│   └── jsrt/
│       ├── images/
│       └── annotations/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   └── model_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── visualization.py
│
├── models/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Dataset Management

The raw datasets are excluded from the Git repository.

Example `.gitignore`:

```gitignore
# Datasets
data/

# Python
__pycache__/
*.pyc

# Jupyter
.ipynb_checkpoints/

# Virtual environment
venv/
.env

# Model files
*.pth
*.pt
*.h5
*.keras
```

If small annotation or metadata files are required for reproducibility, they can be added separately.

---

# Role of Each Dataset

The three datasets provide complementary information for the project.

### VinBigData

**Main role:**

> Abnormality detection and localization

```text
Chest X-ray
     ↓
Detection Model
     ↓
Bounding Boxes
     ↓
Abnormality Localization
```

### ChestX-ray14 + CLAHE

**Main role:**

> Image enhancement and classification

```text
Chest X-ray
     ↓
CLAHE Enhancement
     ↓
Classification Model
     ↓
Pathology Prediction
```

### JSRT

**Main role:**

> External validation and pulmonary nodule analysis

```text
Training
   ↓
Model
   ↓
JSRT
   ↓
External Validation
```

---

# Reproducibility

Before running the notebooks or training scripts:

1. Download the required datasets from their respective sources.
2. Place them in the corresponding directories under `data/`.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Verify the dataset paths in the notebooks or configuration files.
5. Run the preprocessing and analysis notebooks.

---

# Important Considerations

The datasets come from different sources and may have differences in:

* Image resolution
* Image format
* Acquisition equipment
* Patient population
* Annotation methodology
* Class distribution
* Image preprocessing

Therefore, direct comparison between datasets should be performed carefully.

External validation on JSRT can help assess the **generalization capability** of models trained on larger datasets.

---

# Medical Disclaimer

These datasets are intended for **research and educational purposes only**.

The models developed using these datasets are not intended to replace professional medical diagnosis or clinical decision-making.

---

# References

* VinBigData Chest X-ray Abnormalities Detection
* NIH ChestX-ray14
* Japanese Society of Radiological Technology (JSRT) Database
* Wang et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*, CVPR, 2017.
* Shiraishi et al., *Development of a digital image database for chest radiographs with and without a lung nodule*, American Journal of Roentgenology, 2000.
