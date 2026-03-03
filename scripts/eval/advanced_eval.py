import argparse
import logging
import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
import pytorch_lightning as pl

try:
    import omegaconf
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf.base import ContainerMetadata, Node
    import typing
    # Add common types to safe globals for PyTorch 2.6+
    torch.serialization.add_safe_globals([
        ListConfig, DictConfig, ContainerMetadata, Node, 
        typing.Any, typing.Dict, typing.List, typing.Tuple,
        np._core.multiarray.scalar,
        np.dtype,
        np._core.multiarray._reconstruct
    ])
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Label Generation ---
def generate_decompensation_labels(icustays_path, admissions_path, obs_window_hours=48, pred_window_hours=24):
    logger.info("Generating Decompensation labels from MIMIC-IV raw data...")
    icu = pd.read_csv(icustays_path, usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime'])
    adm = pd.read_csv(admissions_path, usecols=['subject_id', 'hadm_id', 'deathtime'])

    icu['intime'] = pd.to_datetime(icu['intime'])
    icu['outtime'] = pd.to_datetime(icu['outtime'])
    adm['deathtime'] = pd.to_datetime(adm['deathtime'])

    df = pd.merge(icu, adm, on=['subject_id', 'hadm_id'], how='left')
    df['icu_duration_hours'] = (df['outtime'] - df['intime']).dt.total_seconds() / 3600
    df = df[df['icu_duration_hours'] >= obs_window_hours].copy()

    df['end_of_observation'] = df['intime'] + pd.Timedelta(hours=obs_window_hours)
    df['end_of_prediction'] = df['end_of_observation'] + pd.Timedelta(hours=pred_window_hours)

    conditions = [
        (df['deathtime'].notnull()) & 
        (df['deathtime'] > df['end_of_observation']) & 
        (df['deathtime'] <= df['end_of_prediction'])
    ]
    df['label_decompensation'] = np.select(conditions, [1], default=0)
    return df[['stay_id', 'label_decompensation']]

def load_and_align_test_data(data_dir, labels_df):
    logger.info(f"Aligning test data from {data_dir}...")
    npz = np.load(os.path.join(data_dir, "test.npz"))
    data = npz['data'].astype(np.float32)
    mask = npz['orig_mask'].astype(np.float32)
    stay_ids = npz['stay_ids']
    
    df_stays = pd.DataFrame({'stay_id': stay_ids, 'idx': np.arange(len(stay_ids))})
    merged = pd.merge(df_stays, labels_df, on='stay_id', how='inner')
    idx = merged['idx'].values
    y = merged['label_decompensation'].values.astype(np.float32)
    
    delta = None
    if 'delta' in npz:
        delta = npz['delta'].astype(np.float32)[idx]
    
    return data[idx], mask[idx], y, delta

# --- Calibration Metric (ECE) ---
def calculate_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

# --- Helper: Run Inference (Impute -> Classify) ---
def run_eval(model, loader, device, permute_idx=None, ablate_kgi=False):
    model.eval()
    all_probs = []
    all_targets = []
    
    # Ablation toggle
    original_kgi = True
    if ablate_kgi and hasattr(model.imputer, 'use_kgi'):
        original_kgi = model.imputer.use_kgi
        model.imputer.use_kgi = False
        
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            y = batch["y_target"].to(device)
            
            # 1. Impute
            # get_imputer_outputs_and_loss returns (imputed, loss)
            imputed, _ = model.get_imputer_outputs_and_loss(batch_dev, is_training=False)
            
            # 2. Permutation (optional)
            if permute_idx is not None:
                perm = torch.randperm(imputed.shape[0])
                imputed[:, :, permute_idx] = imputed[perm, :, permute_idx]
            
            # 3. Classify
            x_modified = torch.cat([imputed, batch_dev["input_mask"]], dim=-1)
            logits = model.classifier(x_modified)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    if ablate_kgi and hasattr(model.imputer, 'use_kgi'):
        model.imputer.use_kgi = original_kgi
        
    return np.concatenate(all_probs), np.concatenate(all_targets)

def get_auprc_score(model, loader, device, permute_idx=None, ablate_kgi=False):
    probs, targets = run_eval(model, loader, device, permute_idx, ablate_kgi)
    return average_precision_score(targets, probs)

# --- Permutation Importance ---
def permutation_importance(model, loader, device):
    model.eval()
    base_score = get_auprc_score(model, loader, device)
    importances = []
    num_features = 17 
    for f_idx in range(num_features):
        logger.info(f"Permuting feature {f_idx}...")
        scores = [get_auprc_score(model, loader, device, permute_idx=f_idx) for _ in range(3)]
        avg_score = np.mean(scores)
        importances.append(base_score - avg_score)
    return np.array(importances), base_score

FEATURE_NAMES = [
    "Heart Rate", "Systolic Blood Pressure", "Diastolic Blood Pressure", 
    "Respiratory Rate", "Oxygen Saturation", "Temperature", "Glucose", 
    "Potassium", "Sodium", "Chloride", "Creatinine", "BUN", 
    "White Blood Cells", "Platelets", "Hemoglobin", "Hematocrit", "Lactate"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--vanilla_ckpt", type=str, required=True)
    parser.add_argument("--kgi_ckpt", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--icustays_path", type=str, default="/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1/icu/icustays.csv.gz")
    parser.add_argument("--admissions_path", type=str, default="/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz")
    parser.add_argument("--output_file", type=str, default="results/advanced_eval_summary.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_importance", action="store_true", help="Skip permutation importance")
    parser.add_argument("--skip_ablation", action="store_true", help="Skip KGI ablation study")
    args = parser.parse_args()
    
    from models.joint.model import JointTrainingModule
    
    labels_df = generate_decompensation_labels(args.icustays_path, args.admissions_path)
    X_test, M_test, y_test, D_test = load_and_align_test_data(args.data_dir, labels_df)
    
    class EvalDataset(torch.utils.data.Dataset):
        def __init__(self, x, m, d, y):
            self.x, self.m, self.y = torch.tensor(x), torch.tensor(m), torch.tensor(y)
            self.d = torch.tensor(d) if d is not None else None
        def __len__(self): return len(self.x)
        def __getitem__(self, i): 
            item = {"data": self.x[i], "input_mask": self.m[i], "orig_mask": self.m[i], 
                    "artificial_mask": torch.zeros_like(self.m[i]), "target": self.x[i], "y_target": self.y[i]}
            if self.d is not None: item["delta"] = self.d[i]
            return item

    loader = DataLoader(EvalDataset(X_test, M_test, D_test, y_test), batch_size=128, shuffle=False)

    results = {}

    def evaluate(ckpt_path, label):
        logger.info(f"Evaluating {label}...")
        model = JointTrainingModule.load_from_checkpoint(ckpt_path, map_location=args.device, strict=False, weights_only=False).to(args.device)
        model.eval()
        
        # A. Basic Metrics
        probs, targets = run_eval(model, loader, args.device)
        auprc = average_precision_score(targets, probs)
        ece = calculate_ece(targets, probs)
        
        # B. Permutation
        imps, _ = permutation_importance(model, loader, args.device)
        res = {
            "auroc": float(roc_auc_score(targets, probs)),
            "auprc": float(auprc),
            "ece": float(ece),
        }
        
        # B. Permutation
        if not args.skip_importance:
            imps, _ = permutation_importance(model, loader, args.device)
            res["importance"] = {name: float(imp) for name, imp in zip(FEATURE_NAMES, imps)}
        
        # C. Ablation for KGI
        if "KGI" in label and not args.skip_ablation: # Use "KGI" to match original label string
            logger.info("Running Knowledge Ablation (re-imputing without KGI)...")
            probs_abl, _ = run_eval(model, loader, args.device, ablate_kgi=True)
            auprc_abl = average_precision_score(targets, probs_abl)
            res["auprc_ablated"] = float(auprc_abl)
            res["kgi_gain"] = float(auprc - auprc_abl)
            
        return res

    results["vanilla"] = evaluate(args.vanilla_ckpt, "Vanilla")
    if args.kgi_ckpt:
        results["kgi"] = evaluate(args.kgi_ckpt, "KGI")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Done. Saved to {args.output_file}")

if __name__ == "__main__":
    main()
