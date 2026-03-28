import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ══════════════════════════════════════════════════════
#  Sesuaikan path
# ══════════════════════════════════════════════════════
LABEL_CSV = "Labels.csv"
PRED_CSV  = "gradcam_results/hasil_prediksi.csv"

# ── Rekonstruksi test set persis seperti saat training ──
# Pakai random_state=42 dan test_size=0.2, sama persis dengan train_kereta.py
df = pd.read_csv(LABEL_CSV)
df["label_numeric"] = df["Label"].map({"GON+": 1, "GON-": 0})

df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=42        # ← sama persis dengan train_kereta.py
)

print(f"Total dataset  : {len(df)}")
print(f"Train set      : {len(df_train)} gambar")
print(f"Test set       : {len(df_test)} gambar")
print(f"  GON+ di test : {df_test['label_numeric'].sum()}")
print(f"  GON- di test : {(df_test['label_numeric']==0).sum()}")

# Buat lookup: image_name → ground truth (hanya test set)
test_lookup = dict(zip(df_test["Image Name"], df_test["label_numeric"]))

# ── Load hasil prediksi, filter hanya gambar test set ──
y_true = []
y_prob = []
skipped = 0

with open(PRED_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        img  = row["image_name"].strip()
        prob = float(row["probability"])

        if img in test_lookup:
            y_true.append(test_lookup[img])
            y_prob.append(prob)
        else:
            skipped += 1

y_true = np.array(y_true)
y_prob = np.array(y_prob)

print(f"\nMatched ke test set : {len(y_true)} gambar")
print(f"Dilewati (train set): {skipped} gambar")

# ── Generate FPR, TPR ──
fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
print(f"\nAUC (test set only) : {roc_auc:.4f}")

for t in [0.5, 0.3]:
    idx = np.argmin(np.abs(thresholds - t))
    pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp_ = confusion_matrix(y_true, pred).ravel()
    print(f"\nThreshold = {t}")
    print(f"  FPR={fpr[idx]:.4f}  TPR={tpr[idx]:.4f}")
    print(f"  TP={tp_}  FP={fp}  FN={fn}  TN={tn}")
    acc = (tp_+tn)/(tp_+fp+fn+tn)
    print(f"  Accuracy={acc*100:.2f}%")

# ── Simpan untuk step 2 ──
np.save("fpr.npy", fpr)
np.save("tpr.npy", tpr)
np.save("thresholds.npy", thresholds)
np.save("y_true.npy", y_true)
np.save("y_prob.npy", y_prob)
print("\n[DONE] .npy tersimpan — jalankan step2_plot_roc.py")
