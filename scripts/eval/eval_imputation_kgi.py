import torch
import pytorch_lightning as pl
from models.saits.model import SAITSModule
from data.dataset import MIMICDataModule
import os
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

# PyTorch 2.6 security: allow OmegaConf types for weight loading
torch.serialization.add_safe_globals([ListConfig, DictConfig, dict])

def evaluate_pair():
    # Final models from yesterday's 50 epoch run
    checkpoints = {
        "SAITS KGI Generic No GNN": "outputs/mimic4/saits/random/2026-02-25_14-58-10/checkpoints/vanilla_KGI/best-epoch=68-val/loss=0.2906.ckpt",
        "SAITS Vanilla": "outputs/mimic4/saits/random/2026-02-24_15-48-19/checkpoints/vanilla/best-epoch=49-val/loss=0.3231.ckpt"
    }

    processed_dir = "data/sota" # Use the SOTA data dir
    
    # Initialize DataModule
    datamodule = MIMICDataModule(
        processed_dir=processed_dir,
        batch_size=128,
        num_workers=4
    )
    datamodule.setup("test")
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1, # Let CUDA_VISIBLE_DEVICES handle it
        precision="16-mixed",
        logger=False
    )

    results = {}
    print("\n" + "="*85)
    print("FINAL COMPARISON: KGI vs VANILLA (SOTA TEST SET)")
    print("="*85)

    for name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"Skipping {name}: checkpoint not found at {ckpt_path}")
            continue
            
        print(f"\nEvaluating {name}...")
        
        # Load model
        # We need to specify use_kgi correctly because the loading heuristic might fail on simple names
        is_kgi = "kgi" in name.lower()
        model = SAITSModule.load_from_checkpoint(
            ckpt_path, 
            use_kgi=is_kgi, 
            strict=False, 
            weights_only=False
        )
        
        # Run test
        test_results = trainer.test(model, datamodule=datamodule, verbose=False)
        results[name] = test_results[0]
        
        # Pull metrics
        m = results[name]
        mae = m.get('test/mae', 0.0)
        rmse = m.get('test/rmse', 0.0)
        mre = m.get('test/mre', 0.0)
        r2 = m.get('test/r2', 0.0)
        print(f"  Result: MAE={mae:.4f}, RMSE={rmse:.4f}, MRE={mre:.4f}, R2={r2:.4f}")

    print("\n" + "="*85)
    print(f"{'Model':<20} | {'MAE ↓':<10} | {'RMSE ↓':<10} | {'MRE ↓':<10} | {'R2 ↑':<10}")
    print("-" * 85)
    
    for name in ["SAITS Vanilla", "SAITS KGI (Previous)", "SAITS KGI High-Fidelity"]:
        if name in results:
            m = results[name]
            print(f"{name:<25} | {m.get('test/mae', 0):<10.4f} | {m.get('test/rmse', 0):<10.4f} | "
                  f"{m.get('test/mre', 0):<10.4f} | {m.get('test/r2', 0):<10.4f}")

if __name__ == "__main__":
    evaluate_pair()
