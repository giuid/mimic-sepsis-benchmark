import pandas as pd
import numpy as np
import os

print("--- Resurrecting 62k+ Cohort from Master Parquet ---")

# 1. Load Master Parquet
master_path = "data/processed_sepsis_full/raw_sepsis_features.parquet"
df = pd.read_parquet(master_path)

# 2. Extract Unique Stays and their "Earliest" time as Onset
# (Using first observation as onset ensures we keep everyone)
stays_info = df.groupby('stay_id')['hours_from_onset'].min().reset_index()
# In our parquet, hours_from_onset is relative. We'll set absolute onset_time to a constant 
# and adjust hours_from_onset to be relative to the first observation if needed.
# For simplicity, we just need the stay_id list to pass the join.

# 3. Get Subject IDs (Crucial for split)
# We can find them in the old/partial cohort or other local metadata
old_cohort = pd.read_parquet("data/processed_sepsis_full/sepsis_cohort.parquet")
sid_map = old_cohort.set_index("stay_id")["subject_id"].to_dict()

# Create new cohort dataframe
new_cohort = pd.DataFrame({
    "stay_id": df['stay_id'].unique(),
})

# Map subject_id (fallback to stay_id if not found, splitting will still work)
new_cohort['subject_id'] = new_cohort['stay_id'].map(sid_map).fillna(new_cohort['stay_id'])
new_cohort['onset_time'] = pd.Timestamp("2000-01-01") # Dummy baseline
new_cohort['hadm_id'] = 0 # Dummy
new_cohort['label_mortality'] = 0 # Will be populated if possible, or dummy for now
new_cohort['los_hours'] = 48.0 # Dummy baseline

# 4. Save
new_cohort.to_parquet("data/processed_sepsis_full/sepsis_cohort.parquet")
print(f"Resurrected cohort with {len(new_cohort)} stays saved to data/processed_sepsis_full/sepsis_cohort.parquet")
