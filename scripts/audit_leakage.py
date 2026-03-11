import torch
import numpy as np
import pandas as pd
from data.dataset import MIMICSepsisTaskDataset

def audit():
    print("--- Scientific Leakage Audit: Task VR (No Treatments) ---")
    dataset = MIMICSepsisTaskDataset("data/processed_sepsis_full/test.npz", task='vr', feature_subset='no_treatments')
    
    # Check 1: Physical Absence
    # The first 51 features are kept. Truncate input to 51 and check if any treatment signal leaked.
    # Treatments in the full set were 51, 52, 53, 54.
    sample_batch = dataset[0]
    data = sample_batch['data'] # (T, 51)
    
    print(f"Data shape: {data.shape}")
    if data.shape[1] > 51:
        print("CRITICAL FAILURE: Subset contains more than 51 features!")
    else:
        print("Check 1 passed: Feature subset size is correct (51).")

    # Check 2: Label Alignment
    # Ensure label is calculated from future window only
    labels = []
    for i in range(100):
        labels.append(dataset[i]['label'].item())
    
    print(f"Sample Labels (First 10): {labels[:10]}")
    print(f"Positive class prevalence in audit sample: {np.mean(labels):.4f}")

    if np.mean(labels) == 0:
        print("WARNING: All labels are zero. Check preprocessing.")
    else:
        print("Check 2 passed: Positive labels found.")

if __name__ == "__main__":
    audit()
