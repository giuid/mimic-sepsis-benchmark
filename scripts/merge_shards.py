import pandas as pd
import glob
import os
from tqdm import tqdm

def merge_shards(pattern, output_name):
    files = glob.glob(pattern)
    print(f"Merging {len(files)} shards for {output_name}...")
    dfs = []
    for f in tqdm(files):
        try:
            # Read CSV from gzip
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_parquet(output_name)
    print(f"Saved {output_name}: {len(full_df)} rows")
    return full_df

# 1. Merge Vitals
vitals_pattern = os.path.expanduser("~/mimic_vitals_extracted/estrazione_vitals/chartevents_*.csv.gz")
merge_shards(vitals_pattern, "data/processed_sepsis_full/raw_vitals.parquet")

# 2. Merge Labs
labs_pattern = os.path.expanduser("~/mimic_labs_extracted/lab/labevents_*.csv.gz")
merge_shards(labs_pattern, "data/processed_sepsis_full/raw_labs.parquet")

print("\n--- Fusion Complete ---")
