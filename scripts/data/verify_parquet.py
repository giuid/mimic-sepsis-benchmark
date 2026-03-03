
import pandas as pd
import os

vitals_path = "data/processed/raw_vitals.parquet"
if os.path.exists(vitals_path):
    df = pd.read_parquet(vitals_path)
    print(f"Vitals: {len(df)} rows")
    print(df.head())
else:
    print("Vitals file not found.")
