import gc
import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (Necessario per IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import root_mean_squared_error

def _free_memory(*objects):
    """
    Forza la pulizia della RAM eliminando i DataFrame temporanei e richiamando il Garbage Collector.
    """
    for obj in objects:
        del obj
    gc.collect()

def load_and_preprocess_data(data_dir: str = "/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1", target_items: list[int] = None) -> pd.DataFrame:
    """
    Carica in modo efficiente (Polars Lazy API) ed elabora i dati di MIMIC-IV, 
    compresi labevents dai file compressi (.csv.gz) originali.
    """
    if target_items is None:
        target_items = [211, 220045, 220179, 220180] # Es. ItemID target per limitare memory explode
        
    # 1. Caricamento Lazy per minimizzare I/O e ottimizzare la RAM (lettura stretta e dtype casting)
    adm_lazy = pl.scan_csv(
        f"{data_dir}/hosp/admissions.csv.gz",
        dtypes={"subject_id": pl.Int32, "hadm_id": pl.Int32}
    ).select(["subject_id", "hadm_id", "admission_type", "insurance"])

    # Lettura ottimizzata filtrando le righe per chartevents
    chart_lazy = pl.scan_csv(
        f"{data_dir}/icu/chartevents.csv.gz",
        dtypes={"subject_id": pl.Int32, "hadm_id": pl.Int32, "itemid": pl.Int32, "valuenum": pl.Float32}
    ).select(["subject_id", "hadm_id", "itemid", "valuenum"]) \
     .filter(pl.col("itemid").is_in(target_items)) \
     .drop_nulls(subset=["hadm_id", "valuenum"]) \
     .group_by(["subject_id", "hadm_id", "itemid"]) \
     .agg(pl.col("valuenum").mean()) # Aggregazione necessaria se vi sono misure multiple per hadm_id

    # Lettura ottimizzata filtrando le righe per labevents (stessa struttura per gli item)
    lab_lazy = pl.scan_csv(
        f"{data_dir}/hosp/labevents.csv.gz",
        dtypes={"subject_id": pl.Int32, "hadm_id": pl.Int32, "itemid": pl.Int32, "valuenum": pl.Float32}
    ).select(["subject_id", "hadm_id", "itemid", "valuenum"]) \
     .filter(pl.col("itemid").is_in(target_items)) \
     .drop_nulls(subset=["hadm_id", "valuenum"]) \
     .group_by(["subject_id", "hadm_id", "itemid"]) \
     .agg(pl.col("valuenum").mean())

    # Concateniamo clinic notes con labevents 
    events_lazy = pl.concat([chart_lazy, lab_lazy])

    # 2. Esecuzione query su RAM (.collect()) e conversione Wide-Format
    df_events = events_lazy.collect().pivot(
        values="valuenum",
        index=["subject_id", "hadm_id"],
        columns="itemid",
        aggregate_function="first"
    )
    
    df_adm = adm_lazy.collect()

    # join in polars e conversione al mondo pandas per Scikit-Learn
    df_merged = df_events.join(df_adm, on=["subject_id", "hadm_id"], how="inner").to_pandas()
    
    # Memory Management: liberiamo esplicitamente Polars dataframes
    _free_memory(df_events, df_adm)

    # 3. Gestione Categoriche (One-Hot Encoding)
    # Gestiamo l'encoding delle covariabili e rimuoviamo gli ID
    df_encoded = pd.get_dummies(
        df_merged.drop(columns=["subject_id", "hadm_id"]), 
        columns=["admission_type", "insurance"], 
        drop_first=True,
        dtype=np.float32 
    )

    # 4. Standardizzazione 
    # Lo StandardScaler è fondamentale in modelli lineari regolarizzati come il LASSO.
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_encoded), 
        columns=df_encoded.columns
    )
    
    return df_scaled


def impute_with_lasso(df: pd.DataFrame, max_features_per_iter: int = 20) -> tuple[pd.DataFrame, dict]:
    """
    Esegue l'imputazione usando un regressore LassoCV.
    Restituisce anche il dizionario delle features selezionate per ogni colonna.
    
    TRUCCO SCALABILITA':
    n_nearest_features: Essenziale su server. Invece di far convergere modelli che usano 
    *tutte* le feature M in un fit O(M^2), limitiamo la regressione iterativa solo ai subset di 
    feature maggiormente correlate per colonna. Interrompe i colli di bottiglia computazionali.
    """
    imputer = IterativeImputer(
        estimator=LassoCV(n_jobs=-1, cv=3, random_state=42), 
        n_nearest_features=max_features_per_iter,
        max_iter=10, 
        random_state=42,
        verbose=1, 
        tol=1e-3,
        keep_empty_features=True
    )
    
    # 1. Calcoliamo l'imputazione
    imputed_data = imputer.fit_transform(df)
    
    # 2. Estrazione delle feature selezionate
    selected_features_dict = {}
    features_names = df.columns.tolist()
    
    # IterativeImputer salva l'estimatore fittato per ogni colonna nel parametro `imputation_sequence_`
    for step in imputer.imputation_sequence_:
        target_idx = step.feat_idx          # L'indice della colonna target di questo step
        target_name = features_names[target_idx]
        
        # Estrattore del modello Lasso per questo target
        lasso_model = step.estimator_
        
        # Quali feature (tra quelle disponibili come predittori) hanno peso != 0?
        # Nota: l'iterative imputer in fit usa gli indici `step.neighbor_feat_idx` come predittori
        predictor_indices = step.neighbor_feat_idx
        non_zero_coef_mask = lasso_model.coef_ != 0
        
        selected_predictor_indices = predictor_indices[non_zero_coef_mask]
        selected_predictor_names = [features_names[i] for i in selected_predictor_indices]
        
        selected_features_dict[target_name] = selected_predictor_names

    print("\n--- Feature Selezionate (LASSO non-zero) ---")
    for target, selected in selected_features_dict.items():
        if selected:
            print(f"Per predire '{target}' LASSO usa: {', '.join([str(v) for v in selected])}")
        else:
            print(f"Per predire '{target}' LASSO ha azzerato tutti i predittori! Usa solo l'intercetta (media).")
    print("--------------------------------------------\n")
            
    return pd.DataFrame(imputed_data, columns=df.columns), selected_features_dict


def validate_imputation(df: pd.DataFrame, missing_rate: float = 0.1):
    """
    Validation Snippet. 
    Maschera artificialmente un set di dati e calcola l'errore d'imputazione (RMSE).
    Si raccomanda di fornire una subset dataframe priva originariamente di NaN (es. prime 10k righe).
    """
    df_complete = df.dropna().copy()
    if df_complete.empty or len(df_complete) < 100:
        print("Dataset di validazione vuoto o troppo piccolo. Necessita subset completamente osservato.")
        return
        
    # Generazione maschera per oscuramento casuale
    np.random.seed(42)
    mask = np.random.rand(*df_complete.shape) < missing_rate
    
    df_masked = df_complete.mask(mask)
    
    # Esecuzione
    df_imputed = impute_with_lasso(df_masked)
    
    # Estraiamo i veri valori pre-maschera e calcoliamo errore d'imputazione
    true_vals = df_complete.values[mask]
    imputed_vals = df_imputed.values[mask]
    
    rmse = root_mean_squared_error(true_vals, imputed_vals)
    print(f"\n[Validation] RMSE su valori mascherati ({missing_rate*100}% mancanti): {rmse:.4f}")


if __name__ == "__main__":
    # WORKFLOW SU SERVER:
    
    DATA_DIR = "/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1"
    # 1. Caricamento Dataframe con lazy polars aggregation
    df_clinico = load_and_preprocess_data(DATA_DIR)
    
    # 2. Ometti qui la validazione per un calcolo diretto
    # validate_imputation(df_clinico)  # <-- Commentato temporaneamente perché la signature è cambiata
    
    # 3. Imputazione Globale del Set (e logging variabili)
    df_imputed, dictionary_of_selections = impute_with_lasso(df_clinico)
    
    # --- SALVATAGGIO RISULTATI ---
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Salva dizionario JSON
    import json
    with open(results_dir / "lasso_feature_selection.json", "w") as f:
        json.dump(dictionary_of_selections, f, indent=4)
        
    # 2. Salva CSV piatto per una rapida occhiata
    rows = []
    for target, selected in dictionary_of_selections.items():
        rows.append({
            "target_variable": target,
            "selected_features": ", ".join([str(v) for v in selected]),
            "n_features": len(selected)
        })
    pd.DataFrame(rows).to_csv(results_dir / "lasso_feature_selection.csv", index=False)
    
    # 3. Salva anche il dataframe imputato (opzionale, ma utile)
    # df_imputed.to_parquet(results_dir / "mimic_imputed_lasso.parquet")
    
    print(f"\n✅ Selezione LASSO completata! Risultati salvati in {results_dir}")
