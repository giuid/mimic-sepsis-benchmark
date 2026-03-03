"""
Analysis: Learned Graph (A_learn) vs UMLS Prior (P)

Extracts the learned adjacency matrices from the "True Prior Nullo" checkpoint
and compares them with the UMLS relational prior to answer:
"Did the model independently discover the connections that exist in UMLS?"

Outputs:
  - Numerical overlap metrics (Jaccard, Cosine, Frobenius)
  - Heatmaps of A_learn, P_UMLS, and their difference
  - Edge-level analysis (which UMLS edges were rediscovered)
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────

# Checkpoint path (True Prior Nullo - best)
CKPT_PATH = "/home/guido/Code/charite/baselines/outputs/saits/random/2026-02-18_15-40-17/checkpoints/best-epoch=97-val/loss=0.3233.ckpt"

# UMLS Relational Prior
PRIOR_PATH = "graph/artifacts_pruned/relational_prior.npy"

# Feature names
FEATURE_NAMES = [
    "Heart Rate", "SBP", "DBP", "Resp Rate", "SpO2", "Temp",
    "Glucose", "K+", "Na+", "Cl-", "Creatinine", "BUN",
    "WBC", "Platelets", "Hgb", "Hct", "Lactate"
]

# Output directory
OUT_DIR = "results/graph_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("ANALYSIS: Learned Graph (A_learn) vs UMLS Prior (P)")
print("=" * 60)

# Load checkpoint
print(f"\nLoading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
state_dict = ckpt["state_dict"]

# Extract all A_learn matrices
a_learn_keys = [k for k in state_dict.keys() if "A_learn" in k]
print(f"Found {len(a_learn_keys)} A_learn matrices:")
for k in a_learn_keys:
    print(f"  - {k}: shape {state_dict[k].shape}")

# Average all A_learn matrices for a single "consensus" graph
A_matrices = [state_dict[k].numpy() for k in a_learn_keys]
A_avg = np.mean(A_matrices, axis=0)
print(f"\nAveraged A_learn shape: {A_avg.shape}")

# Load UMLS Prior
print(f"Loading UMLS Prior: {PRIOR_PATH}")
P = np.load(PRIOR_PATH)
print(f"P shape: {P.shape}")

# ──────────────────────────────────────────────
# 2. NORMALIZE & BINARIZE
# ──────────────────────────────────────────────

# Normalize A_learn to [0, 1] range for comparison
# A_learn can have negative values (it's a bias term), so we use sigmoid
A_norm = 1 / (1 + np.exp(-A_avg))  # Sigmoid normalization

# P is already in [0, 1] range (it's a prior)
P_norm = P.copy()

# Binarize with thresholds for edge overlap analysis
# For A_learn: threshold at median or 0.5 (post-sigmoid)
A_thresh = 0.5
P_thresh = 0.1  # P is sparse, so lower threshold

A_binary = (A_norm > A_thresh).astype(float)
P_binary = (P_norm > P_thresh).astype(float)

# Remove self-loops for edge analysis
np.fill_diagonal(A_binary, 0)
np.fill_diagonal(P_binary, 0)

print(f"\n--- Binarized Graphs ---")
print(f"A_learn edges (thresh={A_thresh}): {int(A_binary.sum())} / {A_binary.size - len(FEATURE_NAMES)} possible")
print(f"P_UMLS  edges (thresh={P_thresh}): {int(P_binary.sum())} / {P_binary.size - len(FEATURE_NAMES)} possible")

# ──────────────────────────────────────────────
# 3. OVERLAP METRICS
# ──────────────────────────────────────────────

# Jaccard Index (on binary edges)
intersection = np.sum(A_binary * P_binary)
union = np.sum(np.clip(A_binary + P_binary, 0, 1))
jaccard = intersection / union if union > 0 else 0

# Recall: Of all UMLS edges, how many did the model rediscover?
umls_edges = P_binary.sum()
recall = intersection / umls_edges if umls_edges > 0 else 0

# Precision: Of all learned edges, how many are in UMLS?
learned_edges = A_binary.sum()
precision = intersection / learned_edges if learned_edges > 0 else 0

# F1
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Continuous metrics (on raw normalized matrices, excluding diagonal)
mask_nodiag = ~np.eye(len(FEATURE_NAMES), dtype=bool)
a_flat = A_norm[mask_nodiag]
p_flat = P_norm[mask_nodiag]

pearson_r, pearson_p = pearsonr(a_flat, p_flat)
spearman_r, spearman_p = spearmanr(a_flat, p_flat)
cosine_sim = np.dot(a_flat, p_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(p_flat) + 1e-8)
frobenius_dist = np.linalg.norm(A_norm - P_norm, 'fro')

print(f"\n{'='*60}")
print(f"OVERLAP METRICS")
print(f"{'='*60}")
print(f"  Jaccard Index:      {jaccard:.4f}")
print(f"  Precision:          {precision:.4f}  (Of learned edges, how many in UMLS?)")
print(f"  Recall:             {recall:.4f}  (Of UMLS edges, how many rediscovered?)")
print(f"  F1 Score:           {f1:.4f}")
print(f"  Intersection:       {int(intersection)} edges")
print(f"  ---")
print(f"  Pearson Corr (r):   {pearson_r:.4f}  (p={pearson_p:.2e})")
print(f"  Spearman Corr (ρ):  {spearman_r:.4f}  (p={spearman_p:.2e})")
print(f"  Cosine Similarity:  {cosine_sim:.4f}")
print(f"  Frobenius Distance: {frobenius_dist:.4f}")

# ──────────────────────────────────────────────
# 4. EDGE-LEVEL ANALYSIS
# ──────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"EDGE-LEVEL ANALYSIS")
print(f"{'='*60}")

# Edges in BOTH A_learn and P
print("\n✅ UMLS edges REDISCOVERED by the model:")
rediscovered = []
for i in range(len(FEATURE_NAMES)):
    for j in range(len(FEATURE_NAMES)):
        if i != j and A_binary[i, j] == 1 and P_binary[i, j] == 1:
            rediscovered.append((i, j, A_norm[i, j], P_norm[i, j]))
            print(f"   {FEATURE_NAMES[i]:>12} → {FEATURE_NAMES[j]:<12} (A={A_norm[i,j]:.3f}, P={P_norm[i,j]:.3f})")

# Edges in P but NOT in A_learn
print("\n❌ UMLS edges MISSED by the model:")
missed = []
for i in range(len(FEATURE_NAMES)):
    for j in range(len(FEATURE_NAMES)):
        if i != j and A_binary[i, j] == 0 and P_binary[i, j] == 1:
            missed.append((i, j, A_norm[i, j], P_norm[i, j]))
            print(f"   {FEATURE_NAMES[i]:>12} → {FEATURE_NAMES[j]:<12} (A={A_norm[i,j]:.3f}, P={P_norm[i,j]:.3f})")

# Edges in A_learn but NOT in P (novel discoveries)
print("\n🔍 NOVEL edges learned (not in UMLS):")
novel = []
for i in range(len(FEATURE_NAMES)):
    for j in range(len(FEATURE_NAMES)):
        if i != j and A_binary[i, j] == 1 and P_binary[i, j] == 0:
            novel.append((i, j, A_norm[i, j], P_norm[i, j]))
            
# Sort by strength and show top 10
novel.sort(key=lambda x: -x[2])
for i, j, a_val, p_val in novel[:15]:
    print(f"   {FEATURE_NAMES[i]:>12} → {FEATURE_NAMES[j]:<12} (A={a_val:.3f})")

print(f"\n   Total novel edges: {len(novel)}")

# ──────────────────────────────────────────────
# 5. VISUALIZATIONS
# ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(28, 6))

# (a) UMLS Prior P
im0 = axes[0].imshow(P_norm, cmap="Blues", vmin=0, vmax=1)
axes[0].set_title("UMLS Prior (P)", fontsize=13, fontweight="bold")
axes[0].set_xticks(range(len(FEATURE_NAMES)))
axes[0].set_yticks(range(len(FEATURE_NAMES)))
axes[0].set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
axes[0].set_yticklabels(FEATURE_NAMES, fontsize=8)
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# (b) Learned A (normalized)
im1 = axes[1].imshow(A_norm, cmap="Reds", vmin=0, vmax=1)
axes[1].set_title("Learned Graph (A_learn, σ-norm)", fontsize=13, fontweight="bold")
axes[1].set_xticks(range(len(FEATURE_NAMES)))
axes[1].set_yticks(range(len(FEATURE_NAMES)))
axes[1].set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
axes[1].set_yticklabels(FEATURE_NAMES, fontsize=8)
plt.colorbar(im1, ax=axes[1], shrink=0.8)

# (c) Difference (A - P)
diff = A_norm - P_norm
max_abs = max(abs(diff.min()), abs(diff.max()))
im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
axes[2].set_title("Difference (A - P)", fontsize=13, fontweight="bold")
axes[2].set_xticks(range(len(FEATURE_NAMES)))
axes[2].set_yticks(range(len(FEATURE_NAMES)))
axes[2].set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
axes[2].set_yticklabels(FEATURE_NAMES, fontsize=8)
plt.colorbar(im2, ax=axes[2], shrink=0.8)

# (d) Overlap (Venn-like binary)
overlap_map = np.zeros_like(A_binary)
for i in range(len(FEATURE_NAMES)):
    for j in range(len(FEATURE_NAMES)):
        if A_binary[i,j] == 1 and P_binary[i,j] == 1:
            overlap_map[i,j] = 3  # Both
        elif A_binary[i,j] == 1:
            overlap_map[i,j] = 2  # Only A
        elif P_binary[i,j] == 1:
            overlap_map[i,j] = 1  # Only P

from matplotlib.colors import ListedColormap
cmap_overlap = ListedColormap(["white", "#3B82F6", "#EF4444", "#8B5CF6"])
im3 = axes[3].imshow(overlap_map, cmap=cmap_overlap, vmin=0, vmax=3)
axes[3].set_title("Edge Overlap", fontsize=13, fontweight="bold")
axes[3].set_xticks(range(len(FEATURE_NAMES)))
axes[3].set_yticks(range(len(FEATURE_NAMES)))
axes[3].set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
axes[3].set_yticklabels(FEATURE_NAMES, fontsize=8)

# Legend for overlap
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color="white", label="No Edge"),
    mpatches.Patch(color="#3B82F6", label="Only UMLS"),
    mpatches.Patch(color="#EF4444", label="Only Learned"),
    mpatches.Patch(color="#8B5CF6", label="Both (Rediscovered)"),
]
axes[3].legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.9)

plt.tight_layout()
save_path = os.path.join(OUT_DIR, "alearn_vs_umls.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\n📊 Saved heatmap to: {save_path}")

# ──────────────────────────────────────────────
# 6. SCATTER PLOT
# ──────────────────────────────────────────────

fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.scatter(p_flat, a_flat, alpha=0.3, s=10, c="#6366F1")
ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label="Perfect Agreement")
ax.set_xlabel("UMLS Prior (P)", fontsize=12)
ax.set_ylabel("Learned Graph (A_learn, σ-norm)", fontsize=12)
ax.set_title(f"Edge Weight Correlation\nPearson r={pearson_r:.3f}, Spearman ρ={spearman_r:.3f}", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

save_path2 = os.path.join(OUT_DIR, "scatter_alearn_vs_umls.png")
plt.savefig(save_path2, dpi=150, bbox_inches="tight")
print(f"📊 Saved scatter to: {save_path2}")

# ──────────────────────────────────────────────
# 7. PER-BLOCK ANALYSIS
# ──────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"PER-BLOCK A_learn ANALYSIS")
print(f"{'='*60}")
for k in a_learn_keys:
    A_k = state_dict[k].numpy()
    A_k_norm = 1 / (1 + np.exp(-A_k))
    A_k_flat = A_k_norm[mask_nodiag]
    r, p = pearsonr(A_k_flat, p_flat)
    print(f"  {k}:")
    print(f"    Pearson r={r:.4f} (p={p:.2e}), Range=[{A_k.min():.3f}, {A_k.max():.3f}]")

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
