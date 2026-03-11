import torch
import os
import numpy as np
from models.joint.sepsis_model import JointSepsisModule
from data.dataset import MIMICDataModule
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path

# CONFIGURATION
checkpoint_path = "outputs/mimic4/joint/random/2026-03-09_14-19-39/checkpoints/default/last.ckpt"

# 1. Load Model
# We expect the gate forcing in layers.py to handle the size mismatch
model = JointSepsisModule.load_from_checkpoint(checkpoint_path)
model.eval()
model.cuda()

# 2. Setup Data
datamodule = MIMICDataModule(
    processed_dir="data/processed_sepsis_full",
    task=model.hparams.task,
    feature_subset="no_treatments",
    batch_size=512,
    num_workers=2
)
datamodule.setup("test")
test_loader = datamodule.test_dataloader()

# 3. Evaluation Loop
all_preds = []
all_targets = []

print(f"Starting evaluation on {len(datamodule.test_dataset)} samples...")

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits, _ = model(batch)
        preds = torch.sigmoid(logits).cpu().numpy()
        targets = batch["label"].squeeze().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets)

# 4. Metrics
auroc = roc_auc_score(all_targets, all_preds)
auprc = average_precision_score(all_targets, all_preds)

print(f"\n--- EVALUATION RESULTS ---")
print(f"Task: VR (No Treatments)")
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")
