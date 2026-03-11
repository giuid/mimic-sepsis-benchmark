import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import os
from omegaconf import OmegaConf

# 1. Setup Credentials
CREDENTIALS_PATH = "mimi-489812-8f16ecf75b4d.json"
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# 2. Load Config for ItemIDs
config_path = "configs/data/mimic4_sepsis_full.yaml"
cfg = OmegaConf.load(config_path)

# Build ItemID Mapping for SQL
itemid_map = {}
for category in ["vitals", "hemodynamics", "labs", "neurological", "respiratory_meta"]:
    for name, ids in cfg[category].items():
        if isinstance(ids, list):
            for i in ids: itemid_map[i] = name
        else:
            itemid_map[ids] = name

# Add treatments
for name, ids in cfg["treatments"]["vasopressors"].items():
    itemid_map[ids[0] if isinstance(ids, list) else ids] = "norepi_equiv"
itemid_map[223848] = "vent_status"
# Fluids
for i in cfg["treatments"]["fluids"]:
    itemid_map[i] = "fluid_volume"

# 3. Load Cohort
cohort_path = "data/processed_sepsis_full/sepsis_cohort.parquet"
cohort = pd.read_parquet(cohort_path)
stay_ids = cohort['stay_id'].tolist()
# BigQuery filter string
stay_ids_str = ",".join([str(s) for s in stay_ids])

print(f"Extracting features for {len(stay_ids)} stays from BigQuery...")

# 4. Comprehensive SQL Query
# We split into ICU and HOSP tables
query = f"""
WITH events AS (
  -- ICU Events (Vitals, GCS, Vent, Vasos)
  SELECT stay_id, charttime, itemid, valuenum as value
  FROM `physionet-data.mimiciv_icu.chartevents`
  WHERE stay_id IN ({stay_ids_str}) 
    AND itemid IN ({",".join(map(str, itemid_map.keys()))})
  
  UNION ALL
  
  -- ICU InputEvents (Vasos, Fluids)
  SELECT stay_id, starttime as charttime, itemid, amount as value
  FROM `physionet-data.mimiciv_icu.inputevents`
  WHERE stay_id IN ({stay_ids_str}) 
    AND itemid IN ({",".join(map(str, itemid_map.keys()))})

  UNION ALL

  -- HOSP LabEvents
  SELECT c.stay_id, le.charttime, le.itemid, le.valuenum as value
  FROM `physionet-data.mimiciv_hosp.labevents` le
  JOIN `physionet-data.mimiciv_icu.icustays` c ON le.hadm_id = c.hadm_id
  WHERE c.stay_id IN ({stay_ids_str})
    AND le.itemid IN ({",".join(map(str, itemid_map.keys()))})
)
SELECT e.*, c.onset_time
FROM events e
JOIN (SELECT stay_id, onset_time FROM `mimi-489812.mimic_derived.sepsis_cohort` ) c ON e.stay_id = c.stay_id
"""

# Note: I assumed you might have a derived table for cohort onset. 
# Since we have it locally, we'll join locally or upload stay_id list.
# To be safe and fast, let's query raw data and align hours locally.

raw_query = f"""
SELECT stay_id, charttime, itemid, valuenum as value
FROM `physionet-data.mimiciv_icu.chartevents`
WHERE stay_id IN UNNEST({stay_ids})
  AND itemid IN ({",".join(map(str, itemid_map.keys()))})
"""

# Actually, BigQuery limits query size. Better approach: use a temporary table or filter by range.
# For this script, we'll use a more robust way: execute in chunks or use a JOIN if cohort is in BQ.

def run_query_and_save(table_name, output_parquet, join_table=None):
    ids_param = bigquery.ArrayQueryParameter("ids", "INT64", stay_ids)
    
    if join_table:
        sql = f"""
        SELECT le.hadm_id as stay_id, le.charttime, le.itemid, le.valuenum as value
        FROM `{table_name}` le
        JOIN `{join_table}` c ON le.hadm_id = c.hadm_id
        WHERE c.stay_id IN UNNEST(@ids)
          AND le.itemid IN ({",".join(map(str, itemid_map.keys()))})
        """
    else:
        sql = f"""
        SELECT stay_id, charttime, itemid, valuenum as value
        FROM `{table_name}`
        WHERE stay_id IN UNNEST(@ids)
          AND itemid IN ({",".join(map(str, itemid_map.keys()))})
        """
    
    job_config = bigquery.QueryJobConfig(query_parameters=[ids_param])
    print(f"Executing query on {table_name}...")
    query_job = client.query(sql, job_config=job_config)
    
    # Use result iterator to save memory
    results = query_job.result()
    df_chunk = results.to_dataframe() # Consider using page iterator if still crashing
    return df_chunk

print("Step 1/3: ICU Chartevents...")
df_ce = run_query_and_save("physionet-data.mimiciv_icu.chartevents", "ce.parquet")

print("Step 2/3: ICU InputEvents...")
df_ie = run_query_and_save("physionet-data.mimiciv_icu.inputevents", "ie.parquet")

print("Step 3/3: HOSP LabEvents...")
df_lab = run_query_and_save("physionet-data.mimiciv_hosp.labevents", "lab.parquet", join_table="physionet-data.mimiciv_icu.icustays")

# Merge
print("Merging and Aligning...")
df = pd.concat([df_ce, df_ie, df_lab])
# Clear memory
del df_ce, df_ie, df_lab

# Alignment logic...
df = pd.merge(df, cohort[['stay_id', 'onset_time']], on='stay_id')
df['feature_name'] = df['itemid'].map(itemid_map)
df['hours_from_onset'] = (pd.to_datetime(df['charttime']) - pd.to_datetime(df['onset_time'])).dt.total_seconds() / 3600.0

output_path = "data/processed_sepsis_full/raw_sepsis_features.parquet"
df[['stay_id', 'hours_from_onset', 'feature_name', 'value']].to_parquet(output_path)
print(f"Extraction complete. Total points: {len(df)}")
