import pandas as pd
import os
import numpy as np

# 1. Paths to local Parquet files
vitals_path = "data/processed_sepsis_full/raw_vitals.parquet"
labs_path = "data/processed_sepsis_full/raw_labs.parquet"
gold_path = "data/processed_sepsis/raw_sepsis_features.parquet" # Has treatments

print("--- Resurrecting Dataset from Local Parquets ---")

# 2. Load
df_v = pd.read_parquet(vitals_path)
df_l = pd.read_parquet(labs_path)
df_g = pd.read_parquet(gold_path)

# 3. Merge and Clean
# We filter gold only for treatments to avoid duplicates with vitals/labs
treatment_names = ["norepi_equiv", "fluid_volume", "urine_output", "vent_status"]
df_t = df_g[df_g['feature_name'].isin(treatment_names)]

print(f"Loaded: Vitals ({len(df_v)}), Labs ({len(df_l)}), Treatments ({len(df_t)})")

df_full = pd.concat([df_v, df_l, df_t]).drop_duplicates()

# 4. Audit
unique_stays = df_full['stay_id'].nunique()
unique_feats = df_full['feature_name'].nunique()
print(f"Audit Result: Unique Stays = {unique_stays}, Unique Features = {unique_feats}")

if unique_feats >= 49:
    print("SUCCESS: Data density is sufficient for the benchmark.")
else:
    print(f"WARNING: Only {unique_feats} features found. Check sources.")

# 5. Save as the 'raw' file for preprocessing
output_path = "data/processed_sepsis_full/raw_sepsis_features.parquet"
df_full.to_parquet(output_path)
print(f"Master Parquet saved to {output_path}")
