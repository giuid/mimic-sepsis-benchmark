import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import os
from pathlib import Path

# Security fix for PyTorch 2.6+
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
import typing
torch.serialization.add_safe_globals([ListConfig, DictConfig, typing.Any, typing.Dict, typing.List, typing.Tuple])

from scripts.evaluate_downstream import impute_with_model, GRUClassifier, load_and_align_data, generate_decompensation_labels

def run_feature_importance_analysis():
    # Setup data
    icustays_path = "/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1/icu/icustays.csv.gz"
    admissions_path = "/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz"
    data_dir = "data/sota"
    
    labels_df = generate_decompensation_labels(icustays_path, admissions_path)
    splits = load_and_align_data(Path(data_dir), labels_df)
    
    # Load feature names (itemids)
    vocab = pd.read_csv("data/embeddings/mimic_vocab_mapped.csv")
    feature_names = vocab['label'].tolist()[:17]
    
    checkpoints = [
        {"name": "Vanilla SAITS", "ckpt": "outputs/mimic4/saits/random/2026-02-24_15-48-19/checkpoints/vanilla/best-epoch=49-val/loss=0.3231.ckpt", "is_kgi": False},
        {"name": "KGI SAITS", "ckpt": "outputs/mimic4/saits/random/2026-02-24_15-48-20/checkpoints/vanilla_KGI/best-epoch=49-val/loss=0.3229.ckpt", "is_kgi": True}
    ]
    
    all_importance = {}
    
    for entry in checkpoints:
        name = entry["name"]
        print(f"\nProcessing {name}...")
        
        # 1. Impute Test Set
        X_test_imputed = impute_with_model(
            "saits_kgi" if entry["is_kgi"] else "saits", 
            entry["ckpt"], 
            splits["test"]
        )
        
        # 2. Train a baseline GRU to measure importance
        print(f"  Training downstream GRU for {name} importance...")
        X_train_imputed = impute_with_model(
            "saits_kgi" if entry["is_kgi"] else "saits", 
            entry["ckpt"], 
            splits["train"]
        )
        X_val_imputed = impute_with_model(
            "saits_kgi" if entry["is_kgi"] else "saits", 
            entry["ckpt"], 
            splits["val"]
        )
        
        pos_weight = (len(splits['train']['y']) - splits['train']['y'].sum()) / splits['train']['y'].sum()
        gru = GRUClassifier(input_dim=17, pos_weight=pos_weight)
        
        import torch.utils.data as data_utils
        train_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train_imputed).float(), torch.from_numpy(splits['train']['y']).float()), batch_size=256, shuffle=True)
        val_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val_imputed).float(), torch.from_numpy(splits['val']['y']).float()), batch_size=256)
        
        import pytorch_lightning as pl
        trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1, logger=False, enable_checkpointing=False)
        trainer.fit(gru, train_loader, val_loader)
        
        gru.eval()
        with torch.no_grad():
            logits = gru(torch.from_numpy(X_test_imputed).float().to(gru.device))
            probs = torch.sigmoid(logits).cpu().numpy()
            base_auprc = average_precision_score(splits['test']['y'], probs)
        
        print(f"  Base AUPRC for {name}: {base_auprc:.4f}")
        
        # 3. Permutation Importance
        drops = []
        for i in range(17):
            X_permuted = X_test_imputed.copy()
            # Permute feature i across all patients but keep the time structure? 
            # Better: permute the entire feature vector across patients
            perm = np.random.permutation(X_permuted.shape[0])
            X_permuted[:, :, i] = X_permuted[perm, :, i]
            
            with torch.no_grad():
                logits_perm = gru(torch.from_numpy(X_permuted).float().to(gru.device))
                probs_perm = torch.sigmoid(logits_perm).cpu().numpy()
                perm_auprc = average_precision_score(splits['test']['y'], probs_perm)
            
            drop = base_auprc - perm_auprc
            drops.append(drop)
            print(f"    - {feature_names[i]}: drop={drop:.4f}")
        
        all_importance[name] = drops

    # Plot Comparison
    plt.figure(figsize=(12, 10))
    x = np.arange(17)
    width = 0.35
    
    plt.bar(x - width/2, all_importance["Vanilla SAITS"], width, label='Vanilla')
    plt.bar(x + width/2, all_importance["KGI SAITS"], width, label='KGI')
    
    plt.ylabel('AUPRC Drop (Importance)')
    plt.title('Feature Importance Comparison: Vanilla vs KGI')
    plt.xticks(x, feature_names, rotation=90)
    plt.legend()
    plt.tight_layout()
    
    out_dir = Path("outputs/importance")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "feature_importance.png")
    
    df_imp = pd.DataFrame(all_importance, index=feature_names)
    df_imp.to_csv(out_dir / "importance_summary.csv")
    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    run_feature_importance_analysis()
