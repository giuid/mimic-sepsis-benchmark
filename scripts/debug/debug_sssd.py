import torch
import numpy as np
import os
import yaml
from models.sssd.model import SSSDModule
from models.sssd.diffusion import p_sample_loop_accelerated
from data.dataset import MIMICDataModule
from torch.utils.data import DataLoader

def debug_check():
    # 1. Load config to get seq_len and features
    processed_dir = "data/processed"
    
    # 2. Setup Data
    dm = MIMICDataModule(
        processed_dir=processed_dir,
        batch_size=8,
        num_workers=0,
        masking_cfg={"type": "random", "p": 0.3}
    )
    dm.setup("test")
    # loader = dm.test_dataloader()
    # batch = next(iter(loader))
    # Synthetic batch for stability test
    batch = {
        "data": torch.randn(8, 48, 17),
        "input_mask": torch.ones(8, 48, 17),
        "artificial_mask": torch.zeros(8, 48, 17)
    }
    
    # 3. Setup Model (Fresh)
    model = SSSDModule(d_feature=17, seq_len=48)
    model.eval()
    model.cuda()
    
    # Move batch to GPU
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    data = batch["data"]
    input_mask = batch["input_mask"]
    observed = data * input_mask
    
    # Track first denoising step
    def debug_model_fn(x_t, t, obs, m):
        out = model.denoiser(x_t, t, obs, m, model.metapath_prior)
        # Check first step of inference
        if t[0] in [999, 49]: 
            print(f"--- Denoiser Call at t={t[0].item()} ---")
            print(f"  Input x_t: mean={x_t.mean().item():.4f}, std={x_t.std().item():.4f}, max={x_t.max().item():.4f}")
            print(f"  Predicted Noise ε: mean={out.mean().item():.4f}, std={out.std().item():.4f}, max={out.max().item():.4f}")
        return out

    # 4. Run Imputation (accelerated)
    print("\nStarting Accelerated Sampling (50 steps)...")
    with torch.no_grad():
        imputed = p_sample_loop_accelerated(
            model_fn=debug_model_fn,
            shape=data.shape,
            schedule=model.schedule,
            observed=observed,
            mask=input_mask,
            device=model.device,
            n_samples=1,
            inference_steps=50
        )
    
    mask = batch["artificial_mask"]
    
    # Ground truth values at imputation positions
    y_true = data[mask == 1].cpu().numpy()
    y_pred = imputed[mask == 1].cpu().numpy()
    
    print("\n=== SSSD Debug Stats ===")
    print(f"Data range: min={data.min().item():.4f}, max={data.max().item():.4f}, mean={data.mean().item():.4f}, std={data.std().item():.4f}")
    print(f"Imputed range: min={imputed.min().item():.4f}, max={imputed.max().item():.4f}, mean={imputed.mean().item():.4f}, std={imputed.std().item():.4f}")
    
    mae = np.abs(y_pred - y_true).mean()
    rmse = np.sqrt(((y_pred - y_true)**2).mean())
    
    print(f"Computed MAE (imputation targets): {mae:.4f}")
    print(f"Computed RMSE (imputation targets): {rmse:.4f}")
    
    # Check if observed values are preserved
    obs_error = torch.abs(imputed[input_mask == 1] - data[input_mask == 1]).mean().item()
    print(f"Observed preservation error: {obs_error:.4e}")
    
    print("\nSample (pred vs true) first 10 values:")
    for p, t in zip(y_pred[:10], y_true[:10]):
        print(f"  Pred: {p:8.4f} | True: {t:8.4f} | Diff: {abs(p-t):8.4f}")

if __name__ == "__main__":
    debug_check()
