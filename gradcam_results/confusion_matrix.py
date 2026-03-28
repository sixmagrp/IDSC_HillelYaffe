import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np

# ══════════════════════════════════════════════════════════════
#  KONFIGURASI — sesuaikan jika data berubah
# ══════════════════════════════════════════════════════════════
TP, FP, FN, TN = 509, 3, 39, 196
AUC = 0.9780

FPR_05, TPR_05 = 0.0151, 0.9279   # operating point threshold=0.5
FPR_03, TPR_03 = 0.0201, 0.9335   # operating point threshold=0.3

# ROC curve (key waypoints — ganti dengan data asli jika tersedia)
FPR_CURVE = [0, 0.001, 0.003, 0.005, 0.008, 0.0151,
             0.02, 0.025, 0.035, 0.05, 0.08,
             0.12, 0.20, 0.35, 0.50, 0.70, 1.0]
TPR_CURVE = [0, 0.72,  0.83,  0.87,  0.905, 0.9279,
             0.9335, 0.944, 0.958, 0.968, 0.975,
             0.981, 0.988, 0.993, 0.996, 0.999, 1.0]

# ══════════════════════════════════════════════════════════════
#  PALET — paper-friendly (grayscale-safe, print-safe)
# ══════════════════════════════════════════════════════════════
C_TP   = '#d6e8d6'   # hijau sangat muda
C_TN   = '#d6e8d6'
C_FN   = '#f2dcd2'   # merah sangat muda
C_FP   = '#faf0d0'   # kuning sangat muda
C_EDGE = '#888888'

C_ROC    = '#333333'   # kurva utama — hitam
C_FILL   = '#cccccc'   # fill AUC — abu-abu
C_RAND   = '#999999'   # diagonal random classifier
C_PT_05  = '#555555'   # titik threshold=0.5 — abu gelap
C_PT_03  = '#111111'   # titik threshold=0.3 — hitam (optimal)
C_ANN_05 = '#444444'
C_ANN_03 = '#111111'

FONT_FAMILY = 'DejaVu Sans'   # fallback; ganti 'Times New Roman' untuk jurnal IEEE/Elsevier

# ══════════════════════════════════════════════════════════════
#  KALKULASI METRIK
# ══════════════════════════════════════════════════════════════
total    = TP + FP + FN + TN
accuracy = (TP + TN) / total
sens     = TP / (TP + FN)
spec     = TN / (TN + FP)
prec     = TP / (TP + FP)
f1       = 2 * prec * sens / (prec + sens)

# ══════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':      FONT_FAMILY,
    'font.size':        9,
    'axes.linewidth':   0.8,
    'axes.edgecolor':   '#444',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
fig.suptitle(
    'GlaucoScan — Model Evaluation on HILLEL-YAFFE Glaucoma Dataset',
    fontsize=11, fontweight='bold', y=1.02, color='#111'
)

# ══════════════════════════════════════════════════════════════
#  PANEL A — Confusion Matrix
# ══════════════════════════════════════════════════════════════
ax1.set_xlim(0, 2); ax1.set_ylim(0, 2.3)
ax1.set_aspect('equal'); ax1.axis('off')

cells = [
    # (col, row, value, label, bg_color)
    (0, 1, TP, 'TP', C_TP),
    (1, 1, FN, 'FN', C_FN),
    (0, 0, FP, 'FP', C_FP),
    (1, 0, TN, 'TN', C_TN),
]

for col, row, val, label, bg in cells:
    rect = FancyBboxPatch(
        (col + 0.07, row + 0.07), 0.86, 0.86,
        boxstyle='round,pad=0.02',
        facecolor=bg, edgecolor=C_EDGE, linewidth=0.7
    )
    ax1.add_patch(rect)
    ax1.text(col + 0.5, row + 0.67, label,
             ha='center', va='center',
             fontsize=10, fontweight='bold', color='#444')
    ax1.text(col + 0.5, row + 0.35, str(val),
             ha='center', va='center',
             fontsize=26, fontweight='bold', color='#111')

# Header labels
ax1.text(0.5, 2.10, 'Predicted GON+', ha='center',
         fontsize=9.5, fontweight='bold', color='#222')
ax1.text(1.5, 2.10, 'Predicted GON−', ha='center',
         fontsize=9.5, fontweight='bold', color='#222')
ax1.text(-0.18, 1.5, 'Actual GON+', ha='center', va='center',
         fontsize=9.5, fontweight='bold', color='#222', rotation=90)
ax1.text(-0.18, 0.5, 'Actual GON−', ha='center', va='center',
         fontsize=9.5, fontweight='bold', color='#222', rotation=90)

# Title
ax1.set_title('(a) Confusion Matrix  (threshold = 0.5)',
              fontsize=10, fontweight='bold', pad=20, color='#111')

# Legend patches
p_correct = mpatches.Patch(facecolor=C_TP, edgecolor=C_EDGE, linewidth=0.7,
                            label='Correct prediction')
p_fn      = mpatches.Patch(facecolor=C_FN, edgecolor=C_EDGE, linewidth=0.7,
                            label='Missed glaucoma (FN)')
p_fp      = mpatches.Patch(facecolor=C_FP, edgecolor=C_EDGE, linewidth=0.7,
                            label='False alarm (FP)')
ax1.legend(handles=[p_correct, p_fn, p_fp],
           loc='lower center', bbox_to_anchor=(0.5, -0.05),
           ncol=1, fontsize=8.5, frameon=True,
           edgecolor='#ccc', facecolor='white')

# Metric summary box
metrics_txt = (
    f'Accuracy = {accuracy*100:.2f}%    Precision = {prec*100:.2f}%\n'
    f'Sensitivity = {sens*100:.2f}%    Specificity = {spec*100:.2f}%\n'
    f'F1-Score = {f1*100:.2f}%    AUC-ROC = {AUC:.4f}'
)
ax1.text(1.0, -0.42, metrics_txt,
         ha='center', va='center', fontsize=8.2, color='#333',
         linespacing=1.6,
         bbox=dict(boxstyle='round,pad=0.45', facecolor='#f8f8f8',
                   edgecolor='#ccc', linewidth=0.7))

# ══════════════════════════════════════════════════════════════
#  PANEL B — ROC Curve
# ══════════════════════════════════════════════════════════════
ax2.set_facecolor('white')

# Curve + fill
ax2.plot(FPR_CURVE, TPR_CURVE,
         color=C_ROC, lw=1.8, zorder=3,
         label=f'ROC curve (AUC = {AUC:.4f})')
ax2.fill_between(FPR_CURVE, TPR_CURVE,
                 alpha=0.10, color=C_FILL, zorder=1)

# Random classifier diagonal
ax2.plot([0, 1], [0, 1],
         color=C_RAND, lw=0.9, linestyle='--',
         label='Random classifier', zorder=2)

# Grid (light, behind everything)
ax2.grid(True, alpha=0.20, linewidth=0.5, color='#bbb', zorder=0)

# ── Operating point threshold = 0.5 ──
ax2.scatter([FPR_05], [TPR_05],
            color=C_PT_05, s=70, zorder=5, marker='o')
ax2.annotate(
    f'thr = 0.5\nFPR={FPR_05:.3f}, TPR={TPR_05:.3f}',
    xy=(FPR_05, TPR_05), xytext=(0.15, 0.77),
    fontsize=8.2, color=C_ANN_05,
    arrowprops=dict(arrowstyle='->', color=C_ANN_05, lw=1.0),
    bbox=dict(boxstyle='round,pad=0.3',
              facecolor='white', edgecolor='#aaa', linewidth=0.7)
)

# ── Operating point threshold = 0.3 (optimal) ──
ax2.scatter([FPR_03], [TPR_03],
            color=C_PT_03, s=85, zorder=5, marker='D')
ax2.annotate(
    f'thr = 0.3  ← optimal screening\nFPR={FPR_03:.3f}, TPR={TPR_03:.3f}',
    xy=(FPR_03, TPR_03), xytext=(0.18, 0.60),
    fontsize=8.2, color=C_ANN_03,
    arrowprops=dict(arrowstyle='->', color=C_ANN_03, lw=1.0),
    bbox=dict(boxstyle='round,pad=0.3',
              facecolor='white', edgecolor='#666', linewidth=0.9,
              linestyle='--')
)

# AUC text label
ax2.text(0.60, 0.12, f'AUC = {AUC:.4f}',
         fontsize=11, fontweight='bold', color='#111',
         bbox=dict(boxstyle='round,pad=0.4',
                   facecolor='#f2f2f2', edgecolor='#999', linewidth=0.8))

# Custom legend
legend_handles = [
    Line2D([0], [0], color=C_ROC, lw=1.8,
           label=f'ROC curve (AUC = {AUC:.4f})'),
    Line2D([0], [0], color=C_RAND, lw=0.9, linestyle='--',
           label='Random classifier'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_PT_05,
           markersize=7, label='Operating point (thr = 0.5)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor=C_PT_03,
           markersize=7, label='Optimal point (thr = 0.3)'),
]
ax2.legend(handles=legend_handles, loc='lower right',
           fontsize=8.5, frameon=True,
           edgecolor='#ccc', facecolor='white')

ax2.set_xlabel('False Positive Rate  (1 − Specificity)', fontsize=9.5, color='#222')
ax2.set_ylabel('True Positive Rate  (Sensitivity / Recall)', fontsize=9.5, color='#222')
ax2.set_title('(b) ROC Curve', fontsize=10, fontweight='bold', pad=12, color='#111')
ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.02)
ax2.tick_params(labelsize=8.5)
for sp in ['top', 'right']:
    ax2.spines[sp].set_visible(False)

# ══════════════════════════════════════════════════════════════
#  SIMPAN
# ══════════════════════════════════════════════════════════════
plt.tight_layout(pad=2.0)
plt.savefig('glaucoscan_eval_paperfriendly.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("[DONE] Tersimpan: glaucoscan_eval_paperfriendly.png")
