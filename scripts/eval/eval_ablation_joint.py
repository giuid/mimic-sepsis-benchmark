import torch
import numpy as np
import os
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from data.dataset import MIMICDataModule
from models.joint.sepsis_model import JointSepsisModule

def evaluate_ablation_checkpoint(ckpt_path, feature_subset, use_kgi):
    print(f"\n--- Evaluating SAITS Joint (96h): Subset={feature_subset}, DKI={use_kgi} ---")
    
    # Setup data
    processed_dir = "data/processed_sepsis_full"
    datamodule = MIMICDataModule(
        processed_dir=processed_dir,
        masking_cfg={"name": "random", "type": "random", "p": 0.3},
        batch_size=64,
        feature_subset=feature_subset,
        task="ihm"
    )
    datamodule.setup()
    test_loader = datamodule.test_dataloader()
    
    d_feature = datamodule.feature_dim
    
    # We need to recreate the args used in JointSepsisModule
    imputator_kwargs = {
        "n_steps": 24,
        "n_features": d_feature,
        "n_layers": 2,
        "d_model": 64,
        "d_inner": 128,
        "n_head": 8,
        "d_k": 8,
        "d_v": 8,
        "dropout": 0.1,
        "use_kgi": use_kgi,
        "kgi_embedding_file": "data/embeddings/medbert_relation_embeddings_sepsis_full.pkl"
    }
    
    model = JointSepsisModule(
        imputator_name="saits",
        imputator_kwargs=imputator_kwargs,
        d_feature=d_feature,
        task="ihm",
        obs_bins=6, # 24h
        feature_indices=datamodule.feature_indices
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits, _ = model(batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
            
    y_prob = np.concatenate(all_probs).flatten()
    y_true = np.concatenate(all_labels).flatten()
    
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    
    print(f"Results: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    return auroc, auprc

if __name__ == "__main__":
    runs = [
        {"ckpt": "outputs/mimic4_sepsis_full/joint/random/2026-03-05_18-50-29/checkpoints/default/best-epoch=21-val/loss=0.1834.ckpt", "subset": "no_treatments", "kgi": False},
        {"ckpt": "outputs/mimic4_sepsis_full/joint/random/2026-03-05_19-04-48/checkpoints/default/best-epoch=16-val/loss=0.1851.ckpt", "subset": "no_treatments", "kgi": True},
        {"ckpt": "outputs/mimic4_sepsis_full/joint/random/2026-03-05_20-08-12/checkpoints/default/best-epoch=15-val/loss=0.1980.ckpt", "subset": "core", "kgi": False},
        {"ckpt": "outputs/mimic4_sepsis_full/joint/random/2026-03-05_20-01-16/checkpoints/default/best-epoch=27-val/loss=0.0576.ckpt", "subset": "core", "kgi": True},
        {"ckpt": "outputs/mimic4_sepsis_full/joint/random/2026-03-05_20-31-06/checkpoints/default/best-epoch=22-val/loss=0.2290.ckpt", "subset": "emergency", "kgi": False},
        {"ckpt": "outputs/mimic4_sepsis_full/joint/random/2026-03-05_20-24-01/checkpoints/default/best-epoch=19-val/loss=0.0664.ckpt", "subset": "emergency", "kgi": True},
    ]
    
    for run in runs:
        if os.path.exists(run["ckpt"]):
            evaluate_ablation_checkpoint(run["ckpt"], run["subset"], run["kgi"])
        else:
            print(f"Missing checkpoint: {run['ckpt']}")
