import polars as pl
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LassoCV

data_dir = "/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1"
target_items = [211, 220045, 220179, 220180]

print("Caricamento subset...")
adm_lazy = pl.scan_csv(f"{data_dir}/hosp/admissions.csv.gz", schema_overrides={"subject_id": pl.Int32, "hadm_id": pl.Int32}).select(["subject_id", "hadm_id", "admission_type", "insurance"]).head(10000)
chart_lazy = pl.scan_csv(f"{data_dir}/icu/chartevents.csv.gz", schema_overrides={"subject_id": pl.Int32, "hadm_id": pl.Int32, "itemid": pl.Int32, "valuenum": pl.Float32}).select(["subject_id", "hadm_id", "itemid", "valuenum"]).filter(pl.col("itemid").is_in(target_items)).drop_nulls(subset=["hadm_id", "valuenum"]).head(1000000)
lab_lazy = pl.scan_csv(f"{data_dir}/hosp/labevents.csv.gz", schema_overrides={"subject_id": pl.Int32, "hadm_id": pl.Int32, "itemid": pl.Int32, "valuenum": pl.Float32}).select(["subject_id", "hadm_id", "itemid", "valuenum"]).filter(pl.col("itemid").is_in(target_items)).drop_nulls(subset=["hadm_id", "valuenum"]).head(1000000)

events_lazy = pl.concat([chart_lazy, lab_lazy])
df_events = events_lazy.collect().pivot(values="valuenum", index=["subject_id", "hadm_id"], columns="itemid", aggregate_function="first")
df_adm = adm_lazy.collect()
df_merged = df_events.join(df_adm, on=["subject_id", "hadm_id"], how="inner").to_pandas()

df_encoded = pd.get_dummies(df_merged.drop(columns=["subject_id", "hadm_id"]), columns=["admission_type", "insurance"], drop_first=True, dtype=float)
df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_encoded), columns=df_encoded.columns)

print(f"Shape: {df_scaled.shape}")
if len(df_scaled) < 5:
    print("Troppi pochi dati nel subset, ignoro LASSO.")
    exit(0)

imputer = IterativeImputer(estimator=LassoCV(n_jobs=-1, cv=3, random_state=42), n_nearest_features=20, max_iter=10, random_state=42, tol=1e-3, keep_empty_features=True)
imputer.fit_transform(df_scaled)

features_names = df_scaled.columns.tolist()
for step in imputer.imputation_sequence_:
    target_name = features_names[step.feat_idx]
    lasso_model = step.estimator_
    selected_predictor_indices = step.neighbor_feat_idx[lasso_model.coef_ != 0]
    selected = [features_names[i] for i in selected_predictor_indices]
    print(f"Per predire '{target_name}' LASSO seleziona: {', '.join([str(v) for v in selected])}")
