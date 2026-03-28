# 👁️ GlaucoScan — Glaucoma Detection System

> Deep learning-based glaucoma detection system with Grad-CAM visualization for model interpretability.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-See%20LICENSE.txt-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📋 Daftar Isi

- [About the Project](#-about-the-project)
- [Feature](#-feature)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [Method of use](#-method-of-use)
- [output and Results](#-output-hasil)
- [Methodology](#-methodology)
- [Contribution](#-contribution)
- [License](#-license)

---

## 🔬 About the Project

**GlaucoScan** is a retinal image classification system for detecting glaucoma using a Convolutional Neural Network (CNN). This model is equipped with:

- **Grad-CAM (Gradient-weighted Class Activation Mapping)** — to visualize the retinal area that is the focus of the model's predictions.
- **Analisis kurva ROC** — for comprehensive evaluation of model performance
- **Confusion matrix** — for classification error analysis
This project aims to help early screening of glaucoma in an automated and explainable manner. (*explainable AI*).

---

## ✨ Feature

| Feature | Description |
|-------|-----------|
| 🧠 **CNN Classification** | Glaucoma detection from retinal fundus images |
| 🔥 **Grad-CAM Visualization** | Heatmap of important areas on retinal images |
| 📊 **ROC/AUC Evaluation** | ROC curve with optimal threshold |
| 📉 **Confusion Matrix** | Visualization of classification performance |
| 📄 **Export Prediction** | Prediction results are saved in CSV format |

---

## 📁 Structure Folder

```
glaucoscan/
│
├── 📁 gradcam_results/              # Grad-CAM visualization results
│   ├── 📁 heatmaps/                 # Heatmap image per sample
│   ├── confusion_matrix.py          # Script generate confusion matrix
│   ├── hasil_prediksi.csv           # Model prediction results
│   ├── ringkasan.txt                # Model performance summary
│   ├── TPRFPR-1.py    # Step 1: calculate FPR & TPR
│   └── ROC-2.py                     # Step 2: plot ROC curve
│
├── 📁 Images/                       # Retinal image dataset
│   └── ...                          # (not pushed, see dataset notes)
│
├── model.py                         # Definition of CNN model architecture
├── Heatmaps-image.py                # Script generate Grad-CAM heatmap
├── Labels.csv                       # Dataset label (file name + class)
├── glaucoscan_eval_final.png        # Final model evaluation graph
├── probability_distribution.png     # Prediction probability distribution
├── fpr.npy                          # False Positive Rate (ROC results)
├── tpr.npy                          # True Positive Rate (ROC results)
├── thresholds.npy                   # Threshold of the ROC curve
├── y_prob.npy                       # Model prediction probability
├── y_true.npy                       # Label ground truth
├── README.md                        # Project documentation (You're here)
└── LICENSE.txt                      # Project license
```

## 🚀 Method of use

### 1. Train/Load Model

```bash
python model.py
```

### 2. Generate Heatmap Grad-CAM

```bash
python Heatmaps-image.py
```
> The heatmap results will be saved in `gradcam_results/heatmaps/`

### 3. Evaluasi Model (ROC Curve)

```bash
# Step 1: Calculate FPR, TPR, and threshold
python gradcam_results/step1_generate_fpr_tpr.py

# Step 2: Plot and save the ROC curve
python gradcam_results/step2.py
```

### 4. Generate Confusion Matrix

```bash
python gradcam_results/confusion_matrix.py
```

### 5. View Prediction Results

Open file `gradcam_results/hasil_prediksi.csv` to see the predictions per image.

---

## 📊 Output & Results

| Output | Location | Description |
|--------|--------|-----------|
| Heatmap Grad-CAM | `gradcam_results/heatmaps/` | Visualization of the model's focus areas |
| Prediction results | `gradcam_results/hasil_prediksi.csv` | Prediction + probability labels |
| Performance summary | `gradcam_results/ringkasan.txt` | Accuracy, AUC, etc. |
| Evaluation graph | `glaucoscan_eval_final.png` | ROC curve & metrics |
| Probability distribution | `probability_distribution.png` | Prediction score distribution |

---

## 🧪 Methodology

```
Retina Image Input
       │
       ▼
  Preprocessing
  (resize, normalization)
       │
       ▼
  Model CNN
  (architecture in model.py)
       │
       ├──► Class Prediction (Glaukoma / Normal)
       │
       └──► Grad-CAM → Heatmap Visualization
```

**Evaluate the model using:**
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC (Area Under Curve)
- Confusion Matrix

---

## 🤝 Contribution

Contributions are very welcome! Please:

1. Fork this repository
2. Create a new branch
3. Commit changes 
4. Push to branch 
5. Create a Pull Request

---

## 📄 License

See the [LICENSE.txt](LICENSE.txt) file for complete license information.

---

## 👤 Author

**Hillel Yaffe**

---

# IDSC_HillelYaffe
