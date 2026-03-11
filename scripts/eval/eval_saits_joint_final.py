import torch
import numpy as np
import os
import hydra
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, average_precision_score
from data.dataset import MIMICDataModule
from models.joint.sepsis_model import JointSepsisModule

def evaluate_joint_checkpoint(ckpt_path, model_name, use_kgi):
    print(f"\n--- Evaluating SAITS Joint: {model_name} (DKI={use_kgi}) ---")
    
    # Setup data
    processed_dir = "data/processed_sepsis_full"
    datamodule = MIMICDataModule(
        processed_dir=processed_dir,
        masking_cfg={"name": "random", "type": "random", "p": 0.3},
        batch_size=32,
    )
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    
    # Load model
    # We need to recreate the args used in JointSepsisModule
    imputator_kwargs = {
        "n_steps": 24,
        "n_features": 55,
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
        d_feature=55,
        task="ihm",
        obs_bins=6 # 24h
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
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
    # SAITS Joint Vanilla
    evaluate_joint_checkpoint(
        "outputs/mimic4_sepsis_full/joint/random/2026-03-04_14-41-13/checkpoints/default/best-epoch=17-val/loss=0.1866.ckpt",
        "SAITS_Joint_Vanilla", False
    )
    # SAITS Joint DKI
    evaluate_joint_checkpoint(
        "outputs/mimic4_sepsis_full/joint/random/2026-03-04_14-53-41/checkpoints/default/best-epoch=17-val/loss=0.1830.ckpt",
        "SAITS_Joint_DKI", True
    )
