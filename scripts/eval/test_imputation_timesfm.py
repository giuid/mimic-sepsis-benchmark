import torch
import numpy as np
from models.timesfm.implementation import TimesFMImputer
import logging

logging.basicConfig(level=logging.INFO)

def test_imputation():
    N, T, D = 2, 48, 3
    data = np.random.randn(N, T, D).astype(np.float32)
    mask = np.ones((N, T, D), dtype=np.float32)
    
    # Introduce some gaps
    mask[0, 10:20, 0] = 0 # Middle gap
    mask[1, 0:5, 1] = 0   # Start gap
    mask[1, 40:48, 2] = 0 # End gap
    
    data = data * mask
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use the official weights ID
    imputer = TimesFMImputer(model_id="google/timesfm-2.5-200m-pytorch", device=device)
    
    imputed = imputer.impute(data, mask)
    
    print(f"Original data shape: {data.shape}")
    print(f"Imputed data shape: {imputed.shape}")
    
    middle_gap_vals = imputed[0, 10:20, 0]
    print(f"Middle gap filled values: {middle_gap_vals}")
    assert not np.all(np.abs(middle_gap_vals) > 1e6) # Sanity check for numerical stability
    
    print("Test imputation complete!")

if __name__ == "__main__":
    test_imputation()
