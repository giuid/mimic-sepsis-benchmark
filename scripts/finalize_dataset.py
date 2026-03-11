import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import os

# 1. Load Config
cfg = OmegaConf.load("configs/data/mimic4_sepsis_full.yaml")

# 2. Build ItemID to FeatureName Mapping (Flattened)
itemid_map = {}
for category in ["vitals", "hemodynamics", "labs", "neurological", "respiratory_meta"]:
    if category in cfg:
        for name, ids in cfg[category].items():
            # Check if it is a list (Hydra ListConfig or standard list)
            if hasattr(ids, "__iter__") and not isinstance(ids, str):
                for i in ids: 
                    itemid_map[int(i)] = name
            else:
                itemid_map[int(ids)] = name

# Add treatments
for name, ids in cfg["treatments"]["vasopressors"].items():
    if hasattr(ids, "__iter__") and not isinstance(ids, str):
        for i in ids: itemid_map[int(i)] = "norepi_equiv"
    else:
        itemid_map[int(ids)] = "norepi_equiv"

for i in cfg["treatments"]["fluids"]:
    itemid_map[int(i)] = "fluid_volume"

itemid_map[223848] = "vent_status"
itemid_map[227443] = "hco3"

print(f"Mapping Dictionary built with {len(itemid_map)} entries.")

# 3. Load Data
vitals = pd.read_parquet("data/processed_sepsis_full/raw_vitals.parquet")
labs = pd.read_parquet("data/processed_sepsis_full/raw_labs.parquet")
# Treatments from Drive (if verified ok)
icu_dir = "data/mimic_drive/3.1/icu/"
input_events = pd.read_csv(os.path.join(icu_dir, "inputevents.csv.gz"), usecols=["stay_id", "starttime", "itemid", "amount"])
output_events = pd.read_csv(os.path.join(icu_dir, "outputevents.csv.gz"), usecols=["stay_id", "charttime", "itemid", "value"])

# 4. Standardize and Merge
print("Standardizing data...")
vitals['itemid'] = vitals['itemid'].astype(int)
labs['itemid'] = labs['itemid'].astype(int)
input_events['itemid'] = input_events['itemid'].astype(int)
output_events['itemid'] = output_events['itemid'].astype(int)

# Use starttime as charttime for input events
input_events.rename(columns={'starttime': 'charttime', 'amount': 'value'}, inplace=True)

df = pd.concat([
    vitals[['stay_id', 'charttime', 'itemid', 'value']],
    labs[['stay_id', 'charttime', 'itemid', 'value']],
    input_events[['stay_id', 'charttime', 'itemid', 'value']],
    output_events[['stay_id', 'charttime', 'itemid', 'value']]
])

# 5. Mapping
print("Mapping features...")
df['feature_name'] = df['itemid'].map(itemid_map)
found = df.dropna(subset=['feature_name'])
print(f"Mapped {len(found)} rows out of {len(df)}. Unique features found: {found['feature_name'].nunique()}")

# 6. Align with Onset
cohort = pd.read_parquet("data/processed_sepsis_full/sepsis_cohort.parquet")
cohort['stay_id'] = cohort['stay_id'].astype(int)
onset_map = cohort.set_index("stay_id")["onset_time"].to_dict()

print("Calculating time bins relative to Sepsis-3 onset...")
found['onset'] = found['stay_id'].map(onset_map)
found = found.dropna(subset=['onset'])
found['hours_from_onset'] = (pd.to_datetime(found['charttime']) - pd.to_datetime(found['onset'])).dt.total_seconds() / 3600.0

# 7. Final Save
output_path = "data/processed_sepsis_full/raw_sepsis_features.parquet"
found[['stay_id', 'hours_from_onset', 'feature_name', 'value']].to_parquet(output_path)
print(f"Dataset resurrected! Final file: {output_path} ({len(found)} points)")
