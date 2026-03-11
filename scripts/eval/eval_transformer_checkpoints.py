import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from data.dataset import MIMICSepsisTaskDataset
from models.sepsis_transformer.model import TimeSeriesTransformer

def extract_all_data(dataset):
    X_list, y_list = [], []
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    for batch in loader:
        X_list.append(batch["data"].numpy())
        y_list.append(batch["label"].numpy())
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

def evaluate_checkpoint(ckpt_path, subset, use_kgi):
    print(f"\n--- Evaluating {ckpt_path} ---")
    data_dir = "data/processed_sepsis_full"
    test_data = MIMICSepsisTaskDataset(os.path.join(data_dir, "test.npz"), task="ihm", feature_subset=subset)
    X_test, y_test = extract_all_data(test_data)
    
    input_dim = X_test.shape[-1]
    model = TimeSeriesTransformer(
        input_dim=input_dim, 
        task_type='classification', 
        use_kgi=use_kgi,
        kgi_embedding_file="data/embeddings/medbert_relation_embeddings_sepsis_full.pkl"
    )
    
    # Correct KGI itemids mapping for the subset
    if use_kgi:
        original_full_itemids = model.kgi_itemids_full
        subset_itemids = [original_full_itemids[i] for i in test_data.feature_indices]
        model.kgi_itemids_full = subset_itemids

    # Load weights
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    preds = model.predict(X_test, batch_size=32)
    auroc = roc_auc_score(y_test, preds)
    auprc = average_precision_score(y_test, preds)
    
    print(f"Results: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    return auroc, auprc

if __name__ == "__main__":
    # Evaluate the two checkpoints we found
    evaluate_checkpoint("checkpoints/sepsis/transformer_ihm_vanilla.pt", "full", False)
    evaluate_checkpoint("checkpoints/sepsis/transformer_ihm_dki.pt", "full", True)
    evaluate_checkpoint("checkpoints/sepsis/transformer_ihm_dki.pt", "no_treatments", True)
