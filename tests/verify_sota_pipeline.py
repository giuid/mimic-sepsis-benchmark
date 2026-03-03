import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocess import build_tensor, filter_valid_stays, split_by_patient, normalize
from data.dataset import ImputationDataset
import torch

def verify_pipeline():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("verify_sota")
    
    # 1. Create Dummy Data
    feature_names = ["heart_rate", "sbp", "glucose", "wbc"] # Subset of 17
    vital_names = ["heart_rate", "sbp"]
    lab_names = ["glucose", "wbc"]
    
    # Create 3 stays:
    # Stay 1: Valid (has vitals and labs)
    # Stay 2: Invalid (only vitals)
    # Stay 3: Invalid (only labs) - actually "At Least One" rule says 1 vital AND 1 lab? 
    #   User said: "Regola 'At Least One': ... se vi è almeno una singola misurazione vitale e almeno una registrazione di laboratorio nell'arco delle 48 ore."
    #   So Stay 2 and 3 should be dropped.
    
    raw_data = [
        # Stay 1: subject 1
        {"stay_id": 101, "subject_id": 1, "hours_since_icu_in": 0.5, "feature_name": "heart_rate", "value": 80},
        {"stay_id": 101, "subject_id": 1, "hours_since_icu_in": 1.5, "feature_name": "glucose", "value": 100},
        
        # Stay 2: subject 2 (Only vitals)
        {"stay_id": 102, "subject_id": 2, "hours_since_icu_in": 0.5, "feature_name": "heart_rate", "value": 75},
        
        # Stay 3: subject 3 (Only labs)
        {"stay_id": 103, "subject_id": 3, "hours_since_icu_in": 2.5, "feature_name": "wbc", "value": 10.5},
        
        # Stay 4: subject 4 (Valid)
        {"stay_id": 104, "subject_id": 4, "hours_since_icu_in": 0.5, "feature_name": "heart_rate", "value": 80},
        {"stay_id": 104, "subject_id": 4, "hours_since_icu_in": 1.5, "feature_name": "glucose", "value": 100},
        
        # Stay 5: subject 5 (Valid)
        {"stay_id": 105, "subject_id": 5, "hours_since_icu_in": 0.5, "feature_name": "sbp", "value": 120},
        {"stay_id": 105, "subject_id": 5, "hours_since_icu_in": 2.5, "feature_name": "wbc", "value": 11.0},
        
        # Stay 6: subject 6 (Valid)
        {"stay_id": 106, "subject_id": 6, "hours_since_icu_in": 1.0, "feature_name": "heart_rate", "value": 70},
        {"stay_id": 106, "subject_id": 6, "hours_since_icu_in": 3.0, "feature_name": "wbc", "value": 9.0},
    ]
    raw_df = pd.DataFrame(raw_data)
    
    # 2. Build Tensor
    T = 5 # Small window
    data, orig_mask, delta, stay_ids, subject_ids = build_tensor(
        raw_df, feature_names, T=T, time_step_hours=1
    )
    
    logger.info("Built tensor shape: %s", data.shape)
    assert data.shape == (6, T, 4)
    assert delta.shape == (6, T, 4)
    
    # Check Delta for Stay 1 (index 0)
    # HR at t=0 (0.5h -> bin 0). Delta at t=0 should be 0 (observed).
    # Glucose at t=1 (1.5h -> bin 1). Delta at t=1 should be 0.
    # HR at t=1 is missing. Delta should be 1.
    # HR at t=2 is missing. Delta should be 2.
    
    # Stay 1 is index 0 (sorted by stay_id usually)
    idx1 = np.where(stay_ids == 101)[0][0]
    
    # HR is feature 0
    # Glucose is feature 2
    
    # t=0, HR: Observed. Delta=0?
    # My code: if observed, delta = (t - last_obs) * step. last_obs starts at -1.
    # t=0: last_obs=-1. delta = 0 - (-1) = 1.
    # So if observed at t=0, delta is 1? 
    #   Standard SAITS: delta is time gap. if first obs is at t=0, gap is from "admission" (-1?).
    #   Usually delta=0 if observed? No, strictly delta is gap. 
    #   If observed, gap from what? From previous obs.
    #   If it's the first obs, gap is from start.
    #   Let's check code logic again:
    #   if observed: delta = (t - last_obs). last_obs becomes t.
    #   So yes, delta[0] = 1.
    #   t=1 (missing): delta = 1 - 0 = 1.
    #   t=2 (missing): delta = 2 - 0 = 2.
    
    print("Delta Stay 1 HR:", delta[idx1, :, 0])
    assert delta[idx1, 0, 0] == 1.0, f"Expected 1.0, got {delta[idx1, 0, 0]}"
    assert delta[idx1, 1, 0] == 1.0, f"Expected 1.0, got {delta[idx1, 1, 0]}"
    assert delta[idx1, 2, 0] == 2.0, f"Expected 2.0, got {delta[idx1, 2, 0]}"
    
    logger.info("Delta validation passed.")
    
    # 3. Filter Valid Stays
    data, orig_mask, delta, stay_ids, subject_ids = filter_valid_stays(
        data, orig_mask, delta, stay_ids, subject_ids,
        feature_names, vital_names, lab_names
    )
    
    logger.info("Filtered tensor shape: %s", data.shape)
    assert len(stay_ids) == 4, f"Expected 4 stays, got {len(stay_ids)}"
    assert stay_ids[0] == 101, f"Expected stay 101, got {stay_ids[0]}"
    
    logger.info("Filtering validation passed.")
    
    # 4. Split
    splits = split_by_patient(
        data, orig_mask, delta, stay_ids, subject_ids,
        train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42
    )
    
    # Since only 1 subject, it will go to one split (likely Train or Test depending on shuffle)
    # Check shape of delta in splits
    for name, sp in splits.items():
        if len(sp["data"]) > 0:
            assert sp["delta"].shape == sp["data"].shape
            logger.info("Split %s has data.", name)
            
    logger.info("Split validation passed.")
    
    # 5. Dataset Delta Computation
    # Save to dummy npz
    np.savez(
        "dummy_train.npz", 
        data=data, orig_mask=orig_mask, delta=delta, 
        stay_ids=stay_ids, subject_ids=subject_ids
    )
    
    ds = ImputationDataset("dummy_train.npz")
    batch = ds[0]
    
    # Case 1: No artificial mask (validation mode)
    # Delta should match saved delta
    assert torch.allclose(batch["delta"], torch.from_numpy(delta[0])), "Dataset delta mismatch (no mask)"
    
    # Case 2: Artificial mask (simulate manual masking)
    # Mask out the HR at t=0
    ds.mask_generator = lambda m, rng: m * 0 # Dummy generator
    # We manually test _compute_delta
    input_mask = batch["orig_mask"].clone()
    input_mask[0, 0] = 0 # HR at t=0 masked
    
    # Recompute delta
    new_delta = ds._compute_delta(input_mask)
    
    # HR at t=0 is now missing.
    # last_obs starts at -1.
    # t=0 (missing): delta = 0 - (-1) = 1.
    # t=1 (missing): delta = 1 - (-1) = 2.
    # t=2 (missing): delta = 2 - (-1) = 3.
    # Original was 1, 1, 2. (Observed at 0).
    
    print("New Delta HR:", new_delta[:, 0])
    assert new_delta[0, 0] == 1.0
    assert new_delta[1, 0] == 2.0
    assert new_delta[2, 0] == 3.0
    
    logger.info("Dataset dynamic delta validation passed.")
    
    # Cleanup
    import os
    os.remove("dummy_train.npz")

if __name__ == "__main__":
    verify_pipeline()
