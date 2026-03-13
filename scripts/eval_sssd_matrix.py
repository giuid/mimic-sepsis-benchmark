import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from models.joint.sepsis_model import JointSepsisModule
from data.dataset import MIMICDataModule

def collect_sssd_checkpoints():
    root_dir = Path("outputs/mimic4/joint/random/")
    results = []
    
    # Iterate through all experiment directories of yesterday and today
    dirs = list(root_dir.glob("2026-03-1[23]_*"))
    dirs.sort()
    
    for exp_dir in dirs:
        meta_file = exp_dir / "experiment_metadata.txt"
        if not meta_file.exists(): continue
        
        # Read task and subset from metadata
        with open(meta_file, "r") as f:
            content = f.read()
            if "imputator_name: sssd" not in content: continue
            
            task = "ihm"
            subset = "full"
            use_kgi = "false"
            
            for line in content.splitlines():
                if "task:" in line: task = line.split(":")[1].strip()
                if "feature_subset:" in line: subset = line.split(":")[1].strip()
                if "use_kgi:" in line: use_kgi = line.split(":")[1].strip()
        
        # Find best checkpoint
        ckpt_dir = exp_dir / "checkpoints" / "default"
        # Look for best epoch subdirs
        best_ckpt_dirs = list(ckpt_dir.glob("best-epoch*"))
        if not best_ckpt_dirs: 
            # Try last.ckpt
            if (ckpt_dir / "last.ckpt").exists():
                best_ckpt = ckpt_dir / "last.ckpt"
            else:
                continue
        else:
            # Sort best directories and pick the one with lowest loss
            best_ckpt_dir = sorted(best_ckpt_dirs)[-1] # Simplified
            ckpts = list(best_ckpt_dir.glob("*.ckpt"))
            if not ckpts: continue
            best_ckpt = ckpts[0]
        
        results.append({
            "task": task,
            "subset": subset,
            "use_kgi": use_kgi,
            "ckpt_path": str(best_ckpt),
            "exp_dir": str(exp_dir)
        })
    
    return results

def evaluate_all():
    checkpoints = collect_sssd_checkpoints()
    print(f"Found {len(checkpoints)} SSSD checkpoints to evaluate.")
    
    final_results = []
    
    # Common masking config
    masking_cfg = {
        "name": "random",
        "type": "random",
        "p": 0.3,
        "eval_seed": 42
    }
    
    for item in checkpoints:
        print(f"\nEvaluating: {item['task']} | {item['subset']} | KGI: {item['use_kgi']}")
        
        try:
            # Load model
            model = JointSepsisModule.load_from_checkpoint(item['ckpt_path'])
            model.eval()
            
            # Setup Data
            dm = MIMICDataModule(
                processed_dir="data/processed_sepsis_full",
                masking_cfg=masking_cfg,
                batch_size=256,
                num_workers=2,
                feature_subset=item['subset'],
                task=item['task']
            )
            
            # Trainer for evaluation
            trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False, enable_progress_bar=False)
            test_results = trainer.test(model, datamodule=dm, verbose=False)
            
            metrics = test_results[0]
            metrics.update(item)
            final_results.append(metrics)
            print(f"Result: AUROC={metrics.get('test/auroc', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"Error evaluating {item['ckpt_path']}: {e}")
            
    # Save to CSV
    if final_results:
        df = pd.DataFrame(final_results)
        df.to_csv("results/sssd_full_evaluation.csv", index=False)
        print("\nEvaluation complete. Results saved to results/sssd_full_evaluation.csv")
    else:
        print("\nNo results collected.")

if __name__ == "__main__":
    evaluate_all()
