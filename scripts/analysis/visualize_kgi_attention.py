import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Fix for PyTorch 2.6+ unpickling issue with omegaconf
try:
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf.base import ContainerMetadata, Node
    import typing
    torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata, Node, typing.Any, typing.Dict, typing.List, typing.Tuple])
except ImportError:
    pass

from models.saits.model import SAITSModule

def load_kgi_model(checkpoint_path: str, d_feature: int = 17, seq_len: int = 48):
    """Loads the completed SAITS KGI model."""
    print(f"Loading KGI model from: {checkpoint_path}")
    
    # We load the weights but we need to explicitly inject the use_kgi kwargs into the base module initialization
    model = SAITSModule.load_from_checkpoint(
        checkpoint_path, 
        d_feature=d_feature, 
        seq_len=seq_len,
        use_kgi=True,
        strict=False
    )
    model.eval()
    return model

def extract_attention_for_patient(model, data_dir: str, patient_idx: int = 0):
    """Runs a single patient through the model and extracts the KGI attention weights."""
    print(f"Loading patient data from {data_dir}...")
    npz_data = np.load(os.path.join(data_dir, "test.npz"))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Extract one patient
    X = torch.tensor(npz_data['data'][patient_idx:patient_idx+1], dtype=torch.float32).to(device)
    M = torch.tensor(npz_data['orig_mask'][patient_idx:patient_idx+1], dtype=torch.float32).to(device)
    
    # We construct a mock batch dictionary as SAITS expects
    batch = {
        "data": X,
        "input_mask": M,
        "artificial_mask": torch.zeros_like(M) # No artificial masking, we want full real observation
    }
    
    print(f"Running Forward Pass for Patient {patient_idx} on {device}...")
    with torch.no_grad():
        _ = model(batch)
        
    # The weights are cached inside the kgi_fusion submodule
    if hasattr(model, 'kgi_fusion') and hasattr(model.kgi_fusion, 'last_attn_weights'):
        attn_matrix = model.kgi_fusion.last_attn_weights.squeeze(0).cpu().numpy() # [T, T]
        return attn_matrix, X.cpu().squeeze(0).numpy(), M.cpu().squeeze(0).numpy()
    else:
        raise ValueError("Model does not have kgi_fusion or last_attn_weights was not cached!")

def plot_attention_heatmap(attn_matrix, mask, save_path: str):
    """Visualizes the temporal cross-attention map highlighting UMLS semantic usage."""
    plt.figure(figsize=(12, 10))
    
    # We want to show how much "Textual Knowledge" at time j was used to impute "Numerical Data" at time i
    seq_len = attn_matrix.shape[0]
    
    sns.heatmap(attn_matrix, cmap="viridis", linewidths=0.05, xticklabels=5, yticklabels=5)
    
    plt.title("KGI Multimodal Cross-Attention Weights (Numerical Query $\\times$ Textual Key)", fontsize=14, pad=20)
    plt.xlabel("UMLS Semantic Context Source (Time Step)", fontsize=12)
    plt.ylabel("Target Time Series Step to Impute (Time Step)", fontsize=12)
    
    # Overlay indicators of where actual observations were
    num_observations_per_step = mask.sum(axis=1)
    
    # Add a secondary axis to show observations
    ax2 = plt.gca().twiny()
    ax2.set_xlim(plt.gca().get_xlim())
    ax2.set_xticks(np.arange(0, seq_len, 5) + 0.5)
    ax2.set_xticklabels([f"Obs: {int(num_observations_per_step[i])}" for i in range(0, seq_len, 5)], rotation=45, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Heatmap successfully saved to: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the KGI .ckpt file")
    parser.add_argument("--data_dir", type=str, default="data/sota", help="Path to Data to pick patient")
    parser.add_argument("--patient_idx", type=int, default=42, help="Test set patient index to visualize")
    args = parser.parse_args()
    
    model = load_kgi_model(args.checkpoint)
    attn_matrix, _, mask = extract_attention_for_patient(model, args.data_dir, args.patient_idx)
    
    # Save the plot
    out_dir = Path("outputs/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"kgi_attention_heatmap_pt{args.patient_idx}.png"
    
    plot_attention_heatmap(attn_matrix, mask, str(plot_path))
