import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import cv2
from datetime import datetime

# KONFIGURASI

MODEL_PATH   = "glaucoma_model.h5"
IMAGES_DIR   = "Images"
OUTPUT_DIR   = "gradcam_results"
HEATMAP_DIR  = os.path.join(OUTPUT_DIR, "heatmaps")
CSV_PATH     = os.path.join(OUTPUT_DIR, "hasil_prediksi.csv")
IMG_SIZE     = (224, 224)
THRESHOLD    = 0.5
# PERSIAPAN FOLDER & MODEL

os.makedirs(HEATMAP_DIR, exist_ok=True)
print(f"[INFO] Output folder: {OUTPUT_DIR}/")

if not os.path.exists(MODEL_PATH):
    sys.exit(f"[ERROR] Model tidak ditemukan: {MODEL_PATH}")

print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Identifikasi convolutional layer terakhir
last_conv_idx  = None
for i, layer in enumerate(model.layers[::-1]):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_idx  = len(model.layers) - 1 - i
        last_conv_name = layer.name
        break

if last_conv_idx is None:
    sys.exit("[ERROR] Tidak ada Conv2D layer ditemukan di model.")

print(f"[INFO] Grad-CAM layer terdeteksi: '{last_conv_name}'")
grad_model_input = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = grad_model_input
conv_output = None

for i, layer in enumerate(model.layers):
    x = layer(x)
    if i == last_conv_idx:
        conv_output = x

grad_model = tf.keras.models.Model(
    inputs=grad_model_input, 
    outputs=[conv_output, x]
)

# FUNGSI GRAD-CAM 
def compute_gradcam_optimized(img_tensor, grad_model):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    
    
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val == 0:
        return heatmap
    return heatmap / max_val

# Batch image executions

all_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
total = len(all_files)
print(f"[INFO] Total gambar ditemukan: {total}\n")

if total == 0:
    sys.exit("[ERROR] Tidak ada gambar di folder Images/")

results    = []
gon_plus   = 0
gon_minus  = 0
start_time = datetime.now()

for idx, fname in enumerate(all_files, 1):
    img_path = os.path.join(IMAGES_DIR, fname)

    try:
        img       = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        x         = tf.expand_dims(img_array, axis=0) 

        preds = model.predict(x, verbose=0)
        prob  = float(preds[0][0])
        label = "GON+" if prob >= THRESHOLD else "GON-"

        heatmap    = compute_gradcam_optimized(x, grad_model)
        
        hm_resized = cv2.resize(heatmap, IMG_SIZE)
        hm_color   = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_INFERNO)
        hm_rgb     = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
        
        img_uint8 = np.uint8(img_array * 255)
        overlay   = cv2.addWeighted(img_uint8, 0.6, hm_rgb, 0.4, 0)

        out_name = fname.rsplit('.', 1)[0] + "_gradcam.png"
        out_path = os.path.join(HEATMAP_DIR, out_name)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"{fname}  |  {label}  (p={prob:.4f})", fontsize=12, fontweight="bold")
        axes[0].imshow(img_array);        axes[0].set_title("Original Fundus"); axes[0].axis("off")
        axes[1].imshow(hm_resized, cmap="inferno"); axes[1].set_title("Activation Heatmap");  axes[1].axis("off")
        axes[2].imshow(overlay);          axes[2].set_title("Clinical Overlay"); axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight") 
        plt.close()

        results.append({"image_name": fname, "probability": round(prob, 6), "label": label, "heatmap_file": out_name})
        if label == "GON+": gon_plus += 1
        else: gon_minus += 1

        if idx % 10 == 0 or idx == total:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"  [{idx:3d}/{total}] {fname:20s}  {label}  p={prob:.4f}  ({elapsed:.1f}s elapsed)")

    except Exception as e:
        print(f"  [SKIP] {fname} — Error: {e}")
        results.append({"image_name": fname, "probability": None, "label": "ERROR", "heatmap_file": ""})


# SAVING CSV FILE
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_name", "probability", "label", "heatmap_file"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n[INFO] CSV tersimpan: {CSV_PATH}")

# STATISTICS SUMMARY

valid_probs = [r["probability"] for r in results if r["probability"] is not None]
errors      = sum(1 for r in results if r["label"] == "ERROR")

print("\n" + "="*50)
print("  RINGKASAN HASIL")
print("="*50)
print(f"  Total gambar diproses : {total}")
print(f"  GON+  (glaucomatous)  : {gon_plus}  ({gon_plus/total*100:.1f}%)")
print(f"  GON-  (non-glaucom.)  : {gon_minus}  ({gon_minus/total*100:.1f}%)")
print(f"  Error / skip          : {errors}")
print(f"  Rata-rata probabilitas: {np.mean(valid_probs):.4f}")
print(f"  Min probabilitas      : {np.min(valid_probs):.4f}")
print(f"  Max probabilitas      : {np.max(valid_probs):.8f}")
print(f"  Waktu total           : {(datetime.now() - start_time).seconds}s")
print("="*50)

# SAVING TXT SUMMARY

summary_path = os.path.join(OUTPUT_DIR, "ringkasan.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"Tanggal         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model           : {MODEL_PATH}\n")
    f.write(f"Total gambar    : {total}\n")
    f.write(f"GON+            : {gon_plus} ({gon_plus/total*100:.1f}%)\n")
    f.write(f"GON-            : {gon_minus} ({gon_minus/total*100:.1f}%)\n")
    f.write(f"Error           : {errors}\n")
    f.write(f"Rata-rata prob  : {np.mean(valid_probs):.4f}\n")
    f.write(f"Min prob        : {np.min(valid_probs):.4f}\n")
    f.write(f"Max prob        : {np.max(valid_probs):.4f}\n")

print(f"\n[INFO] Ringkasan tersimpan: {summary_path}")
print(f"[INFO] Semua heatmap tersimpan di: {HEATMAP_DIR}/")
print("\n[DONE] Batch processing selesai!")