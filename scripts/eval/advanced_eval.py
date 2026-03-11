import argparse
import os
import torch
import warnings
from torchmetrics.classification import BinaryCalibrationError
from torch.utils.data import DataLoader
from data.dataset import MIMICSepsisTaskDataset
from models.joint.sepsis_model import JointSepsisModule
import pandas as pd

warnings.filterwarnings("ignore")

def compute_ece(model, dataloader, device):
    """Computes the Expected Calibration Error (ECE) via BinaryCalibrationError using L1 norm."""
    ece_metric = BinaryCalibrationError(n_bins=10, norm='l1').to(device)
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            # Move relevant tensors to device
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            logits = model(batch_gpu)
            preds = torch.sigmoid(logits)
            targets = batch_gpu['label'].squeeze().int()
            
            ece_metric.update(preds, targets)
            
    ece = ece_metric.compute()
    return ece.item()

def main():
    parser = argparse.ArgumentParser(description="Advanced Sepsis Model Evaluation")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    # Setup Data (using IHM Test set as baseline for calibration)
    print("Loading IHM Test Dataset for Calibration Evaluation...")
    data_dir = os.path.expanduser("~/Data/indus_data/sepsis")
    test_ds = MIMICSepsisTaskDataset(data_dir=data_dir, task="ihm", split="test")
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)
    
    models_to_test = [
        ("SAITS Vanilla", "checkpoints/sepsis/sepsis_ihm_saits_vanilla"),
        ("SAITS KGI",     "checkpoints/sepsis/sepsis_ihm_saits_kgi")
    ]
    
    results = []
    for model_name, ckpt_dir in models_to_test:
        checkpoint_path = os.path.expanduser(f"~/Code/charite/baselines/{ckpt_dir}")
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {model_name}, path does not exist: {checkpoint_path}")
            continue
            
        # Find best checkpoint
        ckpt_files = [f for f in os.listdir(checkpoint_path) if f.startswith("best-epoch")]
        if not ckpt_files:
            continue
        best_ckpt = os.path.join(checkpoint_path, ckpt_files[0])
        
        try:
            model = JointSepsisModule.load_from_checkpoint(best_ckpt)
            model.to(args.device)
            ece = compute_ece(model, test_loader, args.device)
            results.append({"Model": model_name, "ECE": ece})
            print(f"{model_name} ECE: {ece:.4f}")
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            
    if results:
        df = pd.DataFrame(results)
        print("\n=== Expected Calibration Error (ECE) ===")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
