import torch
import numpy as np
import os
import glob
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error
from data.dataset import MIMICDataModule, MIMICSepsisTaskDataset
from models.joint.sepsis_model import JointSepsisModule

def parse_metadata(filepath):
    task = subset = kgi = ckpt = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("task:"):
                task = line.split("task:")[1].strip()
            elif line.startswith("feature_subset:"):
                subset = line.split("feature_subset:")[1].strip()
            elif line.startswith("use_kgi:"):
                kgi = line.split("use_kgi:")[1].strip().lower() == "true"
            elif line.startswith("Best Checkpoint:"):
                ckpt = line.split("Best Checkpoint:")[1].strip()
    return task, subset, kgi, ckpt

def evaluate_run(task, feature_subset, use_kgi, ckpt_path):
    print(f"\n[{task.upper()}] Subset={feature_subset}, KGI={use_kgi}")
    if not os.path.exists(ckpt_path):
        print(f"  -> ERROR: Checkpoint not found at {ckpt_path}")
        return None

    processed_dir = "data/processed_sepsis_full"

    # VR/SS require MIMICSepsisTaskDataset which correctly computes dynamic labels
    if task in ['vr', 'ss']:
        test_ds = MIMICSepsisTaskDataset(
            os.path.join(processed_dir, "test.npz"),
            task=task,
            feature_subset=feature_subset
        )
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
        d_feature = test_ds.data.shape[2]
        feature_indices = test_ds.feature_indices
    else:
        # IHM / LOS: use MIMICDataModule
        datamodule = MIMICDataModule(
            processed_dir=processed_dir,
            masking_cfg={"name": "random", "type": "random", "p": 0.3},
            batch_size=64,
            feature_subset=feature_subset,
            task=task
        )
        datamodule.setup()
        test_loader = datamodule.test_dataloader()
        d_feature = datamodule.feature_dim
        feature_indices = datamodule.feature_indices

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
        task=task,
        obs_bins=6,
        feature_indices=feature_indices
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits, _ = model(batch)
            preds = logits if task == "los" else torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())

    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_labels).flatten()

    results = {}
    if task == "los":
        results["MAE"] = mean_absolute_error(y_true, y_pred)
        results["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"  -> MAE={results['MAE']:.4f}, RMSE={results['RMSE']:.4f}")
    else:
        if len(np.unique(y_true)) < 2:
            print(f"  -> WARNING: No positive samples in test split. Skipping.")
            return None
        results["AUROC"] = roc_auc_score(y_true, y_pred)
        results["AUPRC"] = average_precision_score(y_true, y_pred)
        print(f"  -> AUROC={results['AUROC']:.4f}, AUPRC={results['AUPRC']:.4f}")

    return results

if __name__ == "__main__":
    base_dir = "outputs/mimic4_sepsis_full/joint/random/"
    pattern = os.path.join(base_dir, "*", "experiment_metadata.txt")
    metadata_files = sorted(glob.glob(pattern))

    print(f"Found {len(metadata_files)} runs across all dates.")

    all_results = []
    for f in metadata_files:
        task, subset, kgi, ckpt = parse_metadata(f)
        if not all([task, subset, kgi is not None, ckpt]):
            continue

        # Fix lost checkpoint paths logged as "Running..." during live training
        if not os.path.exists(ckpt):
            run_dir = os.path.dirname(os.path.dirname(os.path.dirname(ckpt)))
            ckpt_dir = os.path.join(run_dir, "checkpoints", "default")
            if os.path.exists(ckpt_dir):
                best_ckpts = [c for c in glob.glob(os.path.join(ckpt_dir, "*.ckpt")) if "best" in c]
                if best_ckpts:
                    ckpt = best_ckpts[0]

        res = evaluate_run(task, subset, kgi, ckpt)
        if res:
            all_results.append({"task": task, "subset": subset, "kgi": kgi, "results": res})

    print("\n\n=== FINAL SUMMARY ===")
    for r in sorted(all_results, key=lambda x: (x["task"], x["subset"], x["kgi"])):
        metrics = ", ".join([f"{k}={v:.4f}" for k, v in r["results"].items()])
        kgi_str = "DKI" if r["kgi"] else "Vanilla"
        print(f"Task: {r['task'].upper():4}, Subset: {r['subset']:16}, Model: SAITS 96h {kgi_str:8} -> {metrics}")
