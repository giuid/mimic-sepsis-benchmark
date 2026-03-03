import torch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import os
from pathlib import Path
import pickle

# Security fix for PyTorch 2.6+
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
import typing
torch.serialization.add_safe_globals([ListConfig, DictConfig, typing.Any, typing.Dict, typing.List, typing.Tuple])

from scripts.evaluate_downstream import impute_with_model, GRUClassifier, load_and_align_data, generate_decompensation_labels
from models.saits.model import SAITSModule

def run_ablation_analysis():
    # Setup data
    icustays_path = "/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1/icu/icustays.csv.gz"
    admissions_path = "/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz"
    data_dir = "data/sota"
    
    labels_df = generate_decompensation_labels(icustays_path, admissions_path)
    splits = load_and_align_data(Path(data_dir), labels_df)
    
    kgi_ckpt = "outputs/mimic4/saits/random/2026-02-24_15-48-20/checkpoints/vanilla_KGI/best-epoch=49-val/loss=0.3229.ckpt"
    
    # 1. Base Results (KGI Full)
    print("\nRunning Base KGI Evaluation...")
    X_test_full = impute_with_model("saits_kgi", kgi_ckpt, splits["test"])
    X_train_full = impute_with_model("saits_kgi", kgi_ckpt, splits["train"])
    X_val_full = impute_with_model("saits_kgi", kgi_ckpt, splits["val"])
    
    pos_weight = (len(splits['train']['y']) - splits['train']['y'].sum()) / splits['train']['y'].sum()
    gru = GRUClassifier(input_dim=17, pos_weight=pos_weight)
    
    import torch.utils.data as data_utils
    train_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_train_full).float(), torch.from_numpy(splits['train']['y']).float()), batch_size=256, shuffle=True)
    val_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_val_full).float(), torch.from_numpy(splits['val']['y']).float()), batch_size=256)
    
    import pytorch_lightning as pl
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1, logger=False, enable_checkpointing=False)
    trainer.fit(gru, train_loader, val_loader)
    
    gru.eval()
    with torch.no_grad():
        logits = gru(torch.from_numpy(X_test_full).float().to(gru.device))
        probs_full = torch.sigmoid(logits).cpu().numpy()
        auprc_full = average_precision_score(splits['test']['y'], probs_full)
    
    print(f"  Base KGI AUPRC: {auprc_full:.4f}")

    # 2. Knowledge Ablation
    # We need to manually load the model and filter its medbert_dict
    print("\nAblating 'Strong' Knowledge...")
    model = SAITSModule.load_from_checkpoint(kgi_ckpt, use_kgi=True, strict=False)
    
    # Define strong itemids (Alarms and Problem List)
    strong_ids = [224640, 220046, 220047, 220056, 220058, 220064, 220066]
    print(f"  Filtering triplets containing IDs: {strong_ids}")
    
    ablated_dict = model.medbert_dict.copy()
    count = 0
    # Ensure strong_ids are integers for matching
    strong_ids = [int(sid) for sid in strong_ids]
    for key in list(ablated_dict.keys()):
        # key is (id_a, id_b)
        if any(int(sid) in [int(k) for k in key] for sid in strong_ids):
            ablated_dict[key] = torch.zeros_like(ablated_dict[key])
            count += 1
    
    print(f"  Zeroed out {count} triplets.")
    model.medbert_dict = ablated_dict
    
    # Run Imputation with ablated model
    class AblatedImputer:
        def __init__(self, model): self.model = model
        def predict(self, dataloaders):
            res = []
            for batch in dataloaders:
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with torch.no_grad():
                    out = self.model(batch)
                    res.append(out["imputed_3"])
            return res

    # Re-run inference using the ablated model inside the imputer logic
    # (We bypass impute_with_model to use our modified instance)
    from torch.utils.data import DataLoader, Dataset
    class SimpleDataset(Dataset):
        def __init__(self, data, mask):
            self.data = torch.from_numpy(data).float()
            self.mask = torch.from_numpy(mask).float()
        def __len__(self): return len(self.data)
        def __getitem__(self, i):
            return {"data": self.data[i], "input_mask": self.mask[i], "artificial_mask": torch.zeros_like(self.mask[i])}

    test_loader = DataLoader(SimpleDataset(splits["test"]["data"], splits["test"]["mask"]), batch_size=256)
    model = model.to("cuda")
    model.eval()
    
    preds_ablated = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            out = model(batch)
            preds_ablated.append(out["imputed_3"].cpu().numpy())
    X_test_ablated_raw = np.concatenate(preds_ablated, axis=0)
    
    # Re-align with observed data
    X_test_ablated = (splits["test"]["mask"] * splits["test"]["data"]) + ((1 - splits["test"]["mask"]) * X_test_ablated_raw)
    
    # Evaluate downstream
    with torch.no_grad():
        logits_ablated = gru(torch.from_numpy(X_test_ablated).float().to(gru.device))
        probs_ablated = torch.sigmoid(logits_ablated).cpu().numpy()
        auprc_ablated = average_precision_score(splits['test']['y'], probs_ablated)
    
    print(f"  Ablated KGI AUPRC: {auprc_ablated:.4f}")
    print(f"  Delta: {auprc_ablated - auprc_full:.4f}")

    # Save results
    out_dir = Path("outputs/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation_results.txt", "w") as f:
        f.write(f"Base AUPRC: {auprc_full}\n")
        f.write(f"Ablated AUPRC: {auprc_ablated}\n")
        f.write(f"Delta: {auprc_ablated - auprc_full}\n")
    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    run_ablation_analysis()
