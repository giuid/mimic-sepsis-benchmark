import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import os
from pathlib import Path

# Security fix for PyTorch 2.6+
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
import typing
torch.serialization.add_safe_globals([ListConfig, DictConfig, typing.Any, typing.Dict, typing.List, typing.Tuple])

from scripts.evaluate_downstream import impute_with_model, GRUClassifier, load_and_align_data, generate_decompensation_labels

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calcola l'ECE."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Needs bin counts to weight the error
    bin_ids = np.digitize(y_prob, np.linspace(0, 1, n_bins + 1)) - 1
    ece = 0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.any(mask):
            bin_prob_true = np.mean(y_true[mask])
            bin_prob_pred = np.mean(y_prob[mask])
            ece += np.sum(mask) * np.abs(bin_prob_true - bin_prob_pred)
    
    return ece / len(y_true)

def run_calibration_analysis():
    # Setup data
    icustays_path = "/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1/icu/icustays.csv.gz"
    admissions_path = "/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz"
    data_dir = "data/sota"
    
    labels_df = generate_decompensation_labels(icustays_path, admissions_path)
    splits = load_and_align_data(Path(data_dir), labels_df)
    
    checkpoints = [
        {"name": "Vanilla SAITS", "ckpt": "outputs/mimic4/saits/random/2026-02-24_15-48-19/checkpoints/vanilla/best-epoch=49-val/loss=0.3231.ckpt", "is_kgi": False},
        {"name": "KGI SAITS", "ckpt": "outputs/mimic4/saits/random/2026-02-24_15-48-20/checkpoints/vanilla_KGI/best-epoch=49-val/loss=0.3229.ckpt", "is_kgi": True}
    ]
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    results = []
    
    for entry in checkpoints:
        name = entry["name"]
        print(f"\nProcessing {name}...")
        
        # 1. Impute Test Set
        X_test_imputed = impute_with_model(
            "saits_kgi" if entry["is_kgi"] else "saits", 
            entry["ckpt"], 
            splits["test"]
        )
        
        # 2. Train/Load a downstream classifier to get probabilities
        # In actual practice, we should use the same classifier used in evaluate_downstream.py
        # Here we re-train a quick one on train_imputed for consistency
        print(f"  Training downstream GRU for {name} calibration...")
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
        
        # Mock GRU Training (simplified)
        pos_weight = (len(splits['train']['y']) - splits['train']['y'].sum()) / splits['train']['y'].sum()
        gru = GRUClassifier(input_dim=17, pos_weight=pos_weight)
        
        import torch.utils.data as data_utils
        train_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train_imputed).float(), torch.from_numpy(splits['train']['y']).float()), batch_size=256, shuffle=True)
        val_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val_imputed).float(), torch.from_numpy(splits['val']['y']).float()), batch_size=256)
        
        import pytorch_lightning as pl
        trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1, logger=False, enable_checkpointing=False)
        trainer.fit(gru, train_loader, val_loader)
        
        # 3. Predict Probabilities on Test
        gru.eval()
        with torch.no_grad():
            logits = gru(torch.from_numpy(X_test_imputed).float().to(gru.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        
        y_true = splits['test']['y']
        
        # 4. Calibration Metrics
        ece = expected_calibration_error(y_true, probs)
        brier = brier_score_loss(y_true, probs)
        
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=f"{name} (ECE={ece:.4f})")
        
        results.append({
            "model": name,
            "ece": ece,
            "brier": brier
        })
        print(f"  {name} Results: ECE={ece:.4f}, Brier={brier:.4f}")

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Diagram: Vanilla vs KGI SAITS")
    plt.legend(loc="lower right")
    
    out_dir = Path("outputs/calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "reliability_diagram.png")
    
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "calibration_summary.csv", index=False)
    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    run_calibration_analysis()
