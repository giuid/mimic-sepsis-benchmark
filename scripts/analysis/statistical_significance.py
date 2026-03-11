import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import resample
from tqdm import tqdm
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

def bootstrap_metrics(y_true, y_prob_a, y_prob_b, n_iterations=1000):
    """
    Perform bootstrap to calculate confidence intervals and p-value for the difference.
    """
    metrics = {
        'auroc_a': [], 'auroc_b': [], 'auroc_diff': [],
        'auprc_a': [], 'auprc_b': [], 'auprc_diff': []
    }
    
    print(f"Running {n_iterations} bootstrap iterations...")
    for i in tqdm(range(n_iterations)):
        # Resample indices with replacement
        indices = resample(np.arange(len(y_true)))
        
        y_t = y_true[indices]
        y_a = y_prob_a[indices]
        y_b = y_prob_b[indices]
        
        # Skip if only one class is present in resample
        if len(np.unique(y_t)) < 2:
            continue
            
        # AUROC
        m_auroc_a = roc_auc_score(y_t, y_a)
        m_auroc_b = roc_auc_score(y_t, y_b)
        metrics['auroc_a'].append(m_auroc_a)
        metrics['auroc_b'].append(m_auroc_b)
        metrics['auroc_diff'].append(m_auroc_b - m_auroc_a)
        
        # AUPRC
        m_auprc_a = average_precision_score(y_t, y_a)
        m_auprc_b = average_precision_score(y_t, y_b)
        metrics['auprc_a'].append(m_auprc_a)
        metrics['auprc_b'].append(m_auprc_b)
        metrics['auprc_diff'].append(m_auprc_b - m_auprc_a)
        
    return metrics

def print_stats(name, values):
    mean = np.mean(values)
    lower = np.percentile(values, 2.5)
    upper = np.percentile(values, 97.5)
    # P-value for difference (proportion of diff <= 0 if we expect b > a)
    p_val = np.mean(np.array(values) <= 0) if mean > 0 else np.mean(np.array(values) >= 0)
    print(f"{name}: {mean:.4f} (95% CI: {lower:.4f} - {upper:.4f}) | p-value: {p_val:.4f}")

def run_analysis():
    data_dir = "data/processed_sepsis_full"
    subset = "no_treatments"
    
    # 1. Load Data
    test_data = MIMICSepsisTaskDataset(os.path.join(data_dir, "test.npz"), task="ihm", feature_subset=subset)
    X_test, y_test = extract_all_data(test_data)
    y_test = y_test.flatten()
    
    # 2. Setup Models
    input_dim = X_test.shape[-1]
    
    # Vanilla Model
    model_v = TimeSeriesTransformer(input_dim=input_dim, use_kgi=False)
    model_v.load_state_dict(torch.load("checkpoints/sepsis/transformer_ihm_vanilla.pt", map_location='cpu'))
    
    # Deep DKI Model
    model_d = TimeSeriesTransformer(input_dim=input_dim, use_kgi=True, 
                                   kgi_embedding_file="data/embeddings/medbert_relation_embeddings_sepsis_full.pkl")
    # Mapping itemids for the subset
    original_full_itemids = model_d.kgi_itemids_full
    subset_itemids = [original_full_itemids[i] for i in test_data.feature_indices]
    model_d.kgi_itemids_full = subset_itemids
    model_d.load_state_dict(torch.load("checkpoints/sepsis/transformer_ihm_dki.pt", map_location='cpu'))
    
    # 3. Predict
    print("Generating predictions...")
    y_prob_v = model_v.predict(X_test, batch_size=32)
    y_prob_d = model_d.predict(X_test, batch_size=32)
    
    # 4. Bootstrap
    results = bootstrap_metrics(y_test, y_prob_v, y_prob_d)
    
    # 5. Report
    print("\n" + "="*40)
    print(f"STATISTICAL ANALYSIS: Vanilla vs Deep DKI ({subset})")
    print("="*40)
    print_stats("AUROC Vanilla", results['auroc_a'])
    print_stats("AUROC Deep DKI", results['auroc_b'])
    print_stats("AUROC Difference", results['auroc_diff'])
    print("-" * 20)
    print_stats("AUPRC Vanilla", results['auprc_a'])
    print_stats("AUPRC Deep DKI", results['auprc_b'])
    print_stats("AUPRC Difference", results['auprc_diff'])
    print("="*40)

if __name__ == "__main__":
    run_analysis()
