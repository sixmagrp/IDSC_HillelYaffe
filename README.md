# 👁️ GlaucoScan — Glaucoma Detection System

> Sistem deteksi glaukoma berbasis deep learning dengan visualisasi Grad-CAM untuk interpretabilitas model.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-See%20LICENSE.txt-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📋 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Fitur](#-fitur)
- [Struktur Folder](#-struktur-folder)
- [Instalasi](#-instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Output & Hasil](#-output--hasil)
- [Metodologi](#-metodologi)
- [Kontribusi](#-kontribusi)
- [Lisensi](#-lisensi)

---

## 🔬 Tentang Proyek

**GlaucoScan** adalah sistem klasifikasi gambar retina untuk mendeteksi glaukoma menggunakan Convolutional Neural Network (CNN). Model ini dilengkapi dengan:

- **Grad-CAM (Gradient-weighted Class Activation Mapping)** — untuk memvisualisasikan area retina yang menjadi fokus prediksi model
- **Analisis kurva ROC** — untuk evaluasi performa model secara komprehensif
- **Confusion matrix** — untuk analisis kesalahan klasifikasi

Proyek ini bertujuan membantu skrining awal glaukoma secara otomatis dan dapat dijelaskan (*explainable AI*).

---

## ✨ Fitur

| Fitur | Deskripsi |
|-------|-----------|
| 🧠 **Klasifikasi CNN** | Deteksi glaukoma dari gambar fundus retina |
| 🔥 **Visualisasi Grad-CAM** | Heatmap area penting pada gambar retina |
| 📊 **Evaluasi ROC/AUC** | Kurva ROC dengan threshold optimal |
| 📉 **Confusion Matrix** | Visualisasi performa klasifikasi |
| 📄 **Export Prediksi** | Hasil prediksi disimpan dalam format CSV |

---

## 📁 Struktur Folder

```
glaucoscan/
│
├── 📁 gradcam_results/              # Hasil visualisasi Grad-CAM
│   ├── 📁 heatmaps/                 # Gambar heatmap per sampel
│   ├── confusion_matrix.py          # Script generate confusion matrix
│   ├── hasil_prediksi.csv           # Hasil prediksi model
│   ├── ringkasan.txt                # Ringkasan performa model
│   ├── TPRFPR-1.py    # Step 1: hitung FPR & TPR
│   └── ROC-2.py                     # Step 2: plot ROC curve
│
├── 📁 Images/                       # Dataset gambar retina
│   └── ...                          # (tidak di-push, lihat catatan dataset)
│
├── model.py                         # Definisi arsitektur model CNN
├── Heatmaps-image.py                # Script generate Grad-CAM heatmap
├── Labels.csv                       # Label dataset (nama file + kelas)
├── glaucoscan_eval_final.png        # Grafik evaluasi akhir model
├── probability_distribution.png     # Distribusi probabilitas prediksi
├── fpr.npy                          # False Positive Rate (hasil ROC)
├── tpr.npy                          # True Positive Rate (hasil ROC)
├── thresholds.npy                   # Threshold dari kurva ROC
├── y_prob.npy                       # Probabilitas prediksi model
├── y_true.npy                       # Label ground truth
├── README.md                        # Dokumentasi proyek (You're here)
└── LICENSE.txt                      # Lisensi proyek
```

## 🚀 Cara Penggunaan

### 1. Latih / Muat Model

```bash
python model.py
```

### 2. Generate Heatmap Grad-CAM

```bash
python Heatmaps-image.py
```
> Hasil heatmap akan tersimpan di `gradcam_results/heatmaps/`

### 3. Evaluasi Model (ROC Curve)

```bash
# Step 1: Hitung FPR, TPR, dan threshold
python gradcam_results/step1_generate_fpr_tpr.py

# Step 2: Plot dan simpan kurva ROC
python gradcam_results/step2.py
```

### 4. Generate Confusion Matrix

```bash
python gradcam_results/confusion_matrix.py
```

### 5. Lihat Hasil Prediksi

Buka file `gradcam_results/hasil_prediksi.csv` untuk melihat prediksi per gambar.

---

## 📊 Output & Hasil

| Output | Lokasi | Deskripsi |
|--------|--------|-----------|
| Heatmap Grad-CAM | `gradcam_results/heatmaps/` | Visualisasi area fokus model |
| Hasil prediksi | `gradcam_results/hasil_prediksi.csv` | Label prediksi + probabilitas |
| Ringkasan performa | `gradcam_results/ringkasan.txt` | Akurasi, AUC, dsb. |
| Grafik evaluasi | `glaucoscan_eval_final.png` | Kurva ROC & metrik |
| Distribusi probabilitas | `probability_distribution.png` | Sebaran skor prediksi |

---

## 🧪 Metodologi

```
Input Gambar Retina
       │
       ▼
  Preprocessing
  (resize, normalisasi)
       │
       ▼
  Model CNN
  (arsitektur di model.py)
       │
       ├──► Prediksi Kelas (Glaukoma / Normal)
       │
       └──► Grad-CAM → Heatmap Visualisasi
```

**Evaluasi model menggunakan:**
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC (Area Under Curve)
- Confusion Matrix

---

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan:

1. Fork repository ini
2. Buat branch baru
3. Commit perubahan 
4. Push ke branch 
5. Buat Pull Request

---

## 📄 Lisensi

Lihat file [LICENSE.txt](LICENSE.txt) untuk informasi lisensi lengkap.

---

## 👤 Author

**Hillel Yaffe**

---

# IDSC_HillelYaffe
