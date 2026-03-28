import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from sklearn.metrics import auc, confusion_matrix

# ══════════════════════════════════════════════════════
#  Load hasil step 1
# ══════════════════════════════════════════════════════
fpr        = np.load("fpr.npy")
tpr        = np.load("tpr.npy")
thresholds = np.load("thresholds.npy")
y_true     = np.load("y_true.npy")
y_prob     = np.load("y_prob.npy")
roc_auc    = auc(fpr, tpr)

prob_pos = y_prob[y_true == 1]
prob_neg = y_prob[y_true == 0]

plt.figure(figsize=(6,4))
plt.hist(prob_pos, bins=40, alpha=0.5, label="GON+", density=True)
plt.hist(prob_neg, bins=40, alpha=0.5, label="GON-", density=True)

plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.title("Probability Distribution")
plt.legend()
plt.grid(alpha=0.2)

plt.savefig("probability_distribution.png", dpi=300)
plt.close()

print("[DEBUG] Saved: probability_distribution.png")

# ── Confusion matrix threshold 0.5 ──
tn, fp, fn, tp = confusion_matrix(y_true, (y_prob >= 0.5).astype(int)).ravel()

# ── Koordinat operating points ──
idx_05  = np.argmin(np.abs(thresholds - 0.5))
fpr_05  = fpr[idx_05];  tpr_05 = tpr[idx_05]

idx_03  = np.argmin(np.abs(thresholds - 0.3))
fpr_03  = fpr[idx_03];  tpr_03 = tpr[idx_03]

# ── Metrik ──
total    = tp + fp + fn + tn
accuracy = (tp + tn) / total
sens     = tp / (tp + fn)
spec     = tn / (tn + fp)
prec     = tp / (tp + fp)
f1       = 2 * prec * sens / (prec + sens)

print(f"AUC         : {roc_auc:.4f}")
print(f"Accuracy    : {accuracy*100:.2f}%")
print(f"Sensitivity : {sens*100:.2f}%")
print(f"Specificity : {spec*100:.2f}%")
print(f"Precision   : {prec*100:.2f}%")
print(f"F1-Score    : {f1*100:.2f}%")
print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"thr=0.5 → FPR={fpr_05:.6f}, TPR={tpr_05:.6f}")
print(f"thr=0.3 → FPR={fpr_03:.6f}, TPR={tpr_03:.6f}")

# ══════════════════════════════════════════════════════
#  Palet — paper-friendly, grayscale-safe
# ══════════════════════════════════════════════════════
C_TP   = '#d6e8d6'
C_TN   = '#d6e8d6'
C_FN   = '#f2dcd2'
C_FP   = '#faf0d0'
C_EDGE = '#888888'

# ══════════════════════════════════════════════════════
#  Figure
# ══════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
fig.suptitle(
    'GlaucoScan — Model Evaluation on HILLEL-YAFFE Glaucoma Dataset',
    fontsize=11, fontweight='bold', y=1.02, color='#111'
)

# ══════════════════════════════════════════════════════
#  Panel A — Confusion Matrix
# ══════════════════════════════════════════════════════
ax1.set_xlim(0, 2); ax1.set_ylim(0, 2.3)
ax1.set_aspect('equal'); ax1.axis('off')

for col, row, val, label, bg in [
    (0, 1, tp, 'TP', C_TP),
    (1, 1, fn, 'FN', C_FN),
    (0, 0, fp, 'FP', C_FP),
    (1, 0, tn, 'TN', C_TN),
]:
    ax1.add_patch(FancyBboxPatch(
        (col+0.07, row+0.07), 0.86, 0.86,
        boxstyle='round,pad=0.02',
        facecolor=bg, edgecolor=C_EDGE, linewidth=0.7
    ))
    ax1.text(col+0.5, row+0.67, label,
             ha='center', va='center',
             fontsize=10, fontweight='bold', color='#444')
    ax1.text(col+0.5, row+0.35, str(val),
             ha='center', va='center',
             fontsize=26, fontweight='bold', color='#111')

ax1.text(0.5,  2.10, 'Predicted GON+', ha='center',
         fontsize=9.5, fontweight='bold', color='#222')
ax1.text(1.5,  2.10, 'Predicted GON−', ha='center',
         fontsize=9.5, fontweight='bold', color='#222')
ax1.text(-0.18, 1.5, 'Actual GON+', ha='center', va='center',
         fontsize=9.5, fontweight='bold', color='#222', rotation=90)
ax1.text(-0.18, 0.5, 'Actual GON−', ha='center', va='center',
         fontsize=9.5, fontweight='bold', color='#222', rotation=90)

ax1.set_title('(a) Confusion Matrix  (threshold = 0.5)',
              fontsize=10, fontweight='bold', pad=20, color='#111')

ax1.legend(
    handles=[
        mpatches.Patch(facecolor=C_TP, edgecolor=C_EDGE, lw=0.7, label='Correct prediction'),
        mpatches.Patch(facecolor=C_FN, edgecolor=C_EDGE, lw=0.7, label='Missed glaucoma (FN)'),
        mpatches.Patch(facecolor=C_FP, edgecolor=C_EDGE, lw=0.7, label='False alarm (FP)'),
    ],
    loc='lower center', bbox_to_anchor=(0.5, -0.05),
    ncol=1, fontsize=8.5, frameon=True,
    edgecolor='#ccc', facecolor='white'
)

ax1.text(1.0, -0.42,
    f'Accuracy={accuracy*100:.2f}%   Precision={prec*100:.2f}%\n'
    f'Sensitivity={sens*100:.2f}%   Specificity={spec*100:.2f}%\n'
    f'F1-Score={f1*100:.2f}%   AUC-ROC={roc_auc:.4f}',
    ha='center', va='center', fontsize=8.2, color='#333', linespacing=1.6,
    bbox=dict(boxstyle='round,pad=0.45', facecolor='#f8f8f8',
              edgecolor='#ccc', linewidth=0.7)
)

# ══════════════════════════════════════════════════════
#  Panel B — ROC Curve
# ══════════════════════════════════════════════════════
ax2.set_facecolor('white')

ax2.plot(fpr, tpr, color='#333333', lw=1.8, zorder=3,
         label=f'ROC curve (AUC = {roc_auc:.4f})')
ax2.fill_between(fpr, tpr, alpha=0.10, color='#cccccc', zorder=1)
ax2.plot([0, 1], [0, 1], color='#999999', lw=0.9,
         linestyle='--', label='Random classifier', zorder=2)
ax2.grid(True, alpha=0.20, linewidth=0.5, color='#bbb', zorder=0)

# Marker threshold 0.5
ax2.scatter([fpr_05], [tpr_05], color='#555555', s=70, zorder=5, marker='o')
ax2.annotate(
    f'thr = 0.5\nFPR={fpr_05:.3f}, TPR={tpr_05:.3f}',
    xy=(fpr_05, tpr_05), xytext=(0.15, 0.77),
    fontsize=8.2, color='#444',
    arrowprops=dict(arrowstyle='->', color='#555', lw=1.0),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              edgecolor='#aaa', linewidth=0.7)
)

# Marker threshold 0.3 (optimal)
ax2.scatter([fpr_03], [tpr_03], color='#111111', s=85, zorder=5, marker='D')
ax2.annotate(
    f'thr = 0.3  ← optimal screening\nFPR={fpr_03:.3f}, TPR={tpr_03:.3f}',
    xy=(fpr_03, tpr_03), xytext=(0.18, 0.60),
    fontsize=8.2, color='#111',
    arrowprops=dict(arrowstyle='->', color='#111', lw=1.0),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              edgecolor='#666', linewidth=0.9, linestyle='--')
)

ax2.text(0.60, 0.12, f'AUC = {roc_auc:.4f}',
         fontsize=11, fontweight='bold', color='#111',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f2f2f2',
                   edgecolor='#999', linewidth=0.8))

ax2.legend(
    handles=[
        Line2D([0],[0], color='#333', lw=1.8,
               label=f'ROC curve (AUC = {roc_auc:.4f})'),
        Line2D([0],[0], color='#999', lw=0.9, linestyle='--',
               label='Random classifier'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#555',
               markersize=7, label='Operating point (thr = 0.5)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#111',
               markersize=7, label='Optimal point (thr = 0.3)'),
    ],
    loc='lower right', fontsize=8.5,
    frameon=True, edgecolor='#ccc', facecolor='white'
)

ax2.set_xlabel('False Positive Rate  (1 − Specificity)', fontsize=9.5, color='#222')
ax2.set_ylabel('True Positive Rate  (Sensitivity / Recall)', fontsize=9.5, color='#222')
ax2.set_title('(b) ROC Curve', fontsize=10, fontweight='bold', pad=12, color='#111')
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.tick_params(labelsize=8.5)
for sp in ['top', 'right']:
    ax2.spines[sp].set_visible(False)

# ══════════════════════════════════════════════════════
#  Simpan
# ══════════════════════════════════════════════════════
plt.tight_layout(pad=2.0)
plt.savefig('glaucoscan_eval_final.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("[DONE] Tersimpan: glaucoscan_eval_final.png")