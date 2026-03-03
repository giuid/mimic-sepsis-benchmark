import torch
import pytorch_lightning as pl
from models.saits.model import SAITSModule
from models.brits.model import BRITSModule
from models.mrnn.model import MRNNModule
from models.gpvae.model import GPVAEModule
from data.dataset import MIMICDataModule
import os
import glob
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

# PyTorch 2.6 security: allow OmegaConf types for weight loading
torch.serialization.add_safe_globals([ListConfig, DictConfig, dict])

def find_best_ckpt(base_dir):
    """Find the latest best-epoch checkpoint in a directory."""
    if not os.path.exists(base_dir):
        return None
    # Find all best checkpoints
    # Structure is usually checkpoints/best-epoch=X-val/loss=Y.ckpt
    ckpts = glob.glob(os.path.join(base_dir, "**/best-epoch*/*.ckpt"), recursive=True)
    if not ckpts:
        # Fallback for flat structure
        ckpts = glob.glob(os.path.join(base_dir, "**/*.ckpt"), recursive=True)
    if not ckpts:
        return None
    # Sort by modification time to get the latest
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]

def evaluate():
    # Directories for SOTA experiments
    output_dirs = {
        "SapBERT": ("saits", "outputs/saits/random/2026-02-19_14-51-00"),
        "BERT (Generic)": ("saits", "outputs/saits/random/2026-02-19_14-51-03"),
        "Random": ("saits", "outputs/saits/random/2026-02-19_14-51-04"),
        "Vanilla": ("saits", "outputs/saits/random/2026-02-19_15-11-55"),
        "SapBERT (Prior Init)": ("saits", "outputs/saits/random/2026-02-19_16-09-02"),
        "BRITS": ("brits", "outputs/brits/random/2026-02-19_15-22-24"),
        "MRNN": ("mrnn", "outputs/mrnn/random/2026-02-19_15-22-28"),
        "GP-VAE": ("gpvae", "outputs/gpvae/random/2026-02-19_15-17-02"),
    }

    model_classes = {
        "saits": SAITSModule,
        "brits": BRITSModule,
        "mrnn": MRNNModule,
        "gpvae": GPVAEModule,
    }

    processed_dir = "data/processed_sota"
    
    # Initialize DataModule
    datamodule = MIMICDataModule(
        processed_dir=processed_dir,
        batch_size=128,
        num_workers=4
    )
    datamodule.setup("test")
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0], # Use GPU 0
        precision="16-mixed",
        logger=False
    )

    results = {}
    print("\n" + "="*85)
    print("COMPREHENSIVE SOTA EVALUATION (TEST SET)")
    print("="*85)

    for name, (m_type, base_dir) in output_dirs.items():
        ckpt_path = find_best_ckpt(base_dir)
        
        if not ckpt_path:
            print(f"Skipping {name}: no checkpoint found in {base_dir}")
            continue
            
        print(f"\nEvaluating {name}...")
        print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
        
        # Load model
        model_class = model_classes[m_type]
        model = model_class.load_from_checkpoint(ckpt_path, weights_only=False)
        
        # Run test
        test_results = trainer.test(model, datamodule=datamodule, verbose=False)
        results[name] = test_results[0]
        
        # Pull metrics
        try:
            m = results[name]
            mae = m.get('test/mae', 0.0)
            rmse = m.get('test/rmse', 0.0)
            mre = m.get('test/mre', 0.0)
            r2 = m.get('test/r2', 0.0)
            corr_err = m.get('test/corr_err', 0.0)
            print(f"  Result: MAE={mae:.4f}, RMSE={rmse:.4f}, MRE={mre:.4f}, R2={r2:.4f}, CorrErr={corr_err:.4f}")
        except Exception as e:
            print(f"  Error calculating metrics for {name}: {e}")

    print("\n" + "="*95)
    print(f"{'Model':<20} | {'MAE ↓':<8} | {'RMSE ↓':<8} | {'MRE ↓':<8} | {'R2 ↑':<8} | {'Corr.Err ↓':<10}")
    print("-" * 95)
    
    # Sort by MAE for readability
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('test/mae', 999))
    
    for name, m in sorted_results:
        print(f"{name:<20} | {m.get('test/mae', 0):<8.4f} | {m.get('test/rmse', 0):<8.4f} | "
              f"{m.get('test/mre', 0):<8.4f} | {m.get('test/r2', 0):<8.4f} | {m.get('test/corr_err', 0):<10.4f}")

if __name__ == "__main__":
    evaluate()
