import argparse
import logging
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Fix for PyTorch 2.6+ unpickling issue with omegaconf
try:
    import omegaconf
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf.base import ContainerMetadata, Node
    import typing
    # Add common types to safe globals for PyTorch 2.6+
    torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata, Node, typing.Any, typing.Dict, typing.List, typing.Tuple])
except ImportError:
    pass


def generate_decompensation_labels(icustays_path, admissions_path, obs_window_hours=48, pred_window_hours=24):
    """
    Genera label per il task di Decompensation.
    Dato un periodo di osservazione iniziale (es. 48h dall'ingresso in ICU),
    il paziente morirà nelle successive `pred_window_hours` (es. 24h)?
    """
    logger.info("Caricamento dataset MIMIC-IV per generazione label Decompensation...")
    icu = pd.read_csv(icustays_path, usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime'])
    adm = pd.read_csv(admissions_path, usecols=['subject_id', 'hadm_id', 'deathtime'])

    icu['intime'] = pd.to_datetime(icu['intime'])
    icu['outtime'] = pd.to_datetime(icu['outtime'])
    adm['deathtime'] = pd.to_datetime(adm['deathtime'])

    df = pd.merge(icu, adm, on=['subject_id', 'hadm_id'], how='left')

    df['icu_duration_hours'] = (df['outtime'] - df['intime']).dt.total_seconds() / 3600
    df = df[df['icu_duration_hours'] >= obs_window_hours].copy()

    df['end_of_observation'] = df['intime'] + pd.Timedelta(hours=obs_window_hours)
    df['end_of_prediction'] = df['end_of_observation'] + pd.Timedelta(hours=pred_window_hours)

    conditions = [
        (df['deathtime'].notnull()) & 
        (df['deathtime'] > df['end_of_observation']) & 
        (df['deathtime'] <= df['end_of_prediction'])
    ]
    df['label_decompensation'] = np.select(conditions, [1], default=0)

    final_labels = df[['stay_id', 'label_decompensation']]
    logger.info(f"Label estratte. Totale record: {len(final_labels)}, Casi positivi: {final_labels['label_decompensation'].sum()}")
    
    return final_labels


def generate_sepsis_outcome_labels(sepsis_cohort_path, pred_window_hours=24):
    """
    Generates outcome labels specifically for the Sepsis Benchmark.
    Uses the pre-calculated sepsis_cohort.parquet which has the 'onset_time'.
    """
    logger.info(f"Generating Sepsis Outcome labels from {sepsis_cohort_path}...")
    cohort = pd.read_parquet(sepsis_cohort_path)
    
    if 'label_mortality' in cohort.columns:
        return cohort[['stay_id', 'label_mortality']].rename(columns={'label_mortality': 'label_decompensation'})

    # Fallback logic if label_mortality is not there
    # Huang et al. 2025: Observation is up to onset + 72h
    cohort['end_of_observation'] = cohort['onset_time'] + pd.Timedelta(hours=72)
    cohort['end_of_prediction'] = cohort['end_of_observation'] + pd.Timedelta(hours=pred_window_hours)
    
    mimic_root = "/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1"
    adm = pd.read_csv(Path(mimic_root) / "hosp" / "admissions.csv.gz", usecols=['hadm_id', 'deathtime'])
    adm['deathtime'] = pd.to_datetime(adm['deathtime'])
    
    cohort['hadm_id'] = cohort['hadm_id'].astype(int)
    adm['hadm_id'] = adm['hadm_id'].astype(int)
    
    df = pd.merge(cohort, adm, on='hadm_id', how='left')
    
    dt_col = 'deathtime' if 'deathtime' in df.columns else ('deathtime_y' if 'deathtime_y' in df.columns else 'deathtime_x')

    conditions = [
        (df[dt_col].notnull()) & 
        (df[dt_col] > df['end_of_observation']) & 
        (df[dt_col] <= df['end_of_prediction'])
    ]
    df['label_outcome'] = np.select(conditions, [1], default=0)
    
    return df[['stay_id', 'label_outcome']].rename(columns={'label_outcome': 'label_decompensation'})


def calculate_delta(mask):
    """Calcola i gap temporali (delta) basandosi sulla maschera delle osservazioni.
    mask: (N, T, D) dove 1 significa osservato, 0 significa mancante.
    Huang et al. 2025 style delta calculation.
    """
    N, T, D = mask.shape
    delta = np.zeros_like(mask)
    for t in range(1, T):
        # Se t-1 era osservato, delta=1. Altrimenti delta=1 + delta_prec
        delta[:, t, :] = 1 + (1 - mask[:, t-1, :]) * delta[:, t-1, :]
    return delta


def load_and_align_data(data_dir: Path, labels_df: pd.DataFrame):
    """Carica i dati NPZ e li allinea con le label."""
    splits = {}
    for split in ['train', 'val', 'test']:
        npz = np.load(data_dir / f"{split}.npz")
        # Array originali
        data = npz['data'].astype(np.float32)
        mask = npz['orig_mask'].astype(np.float32)
        stay_ids = npz['stay_ids']
        
        # Merge con le label
        df_stays = pd.DataFrame({'stay_id': stay_ids, 'idx': np.arange(len(stay_ids))})
        merged = pd.merge(df_stays, labels_df, on='stay_id', how='inner')
        
        idx = merged['idx'].values
        y = merged['label_decompensation'].values.astype(np.float32)
        
        if 'delta' in npz:
            delta_val = npz['delta'].astype(np.float32)[idx]
        else:
            logger.info(f"Delta missing for {split}, calculating from mask...")
            delta_val = calculate_delta(mask[idx])

        splits[split] = {
            'data': data[idx],
            'mask': mask[idx],
            'delta': delta_val,
            'y': y
        }
        logger.info(f"Split {split}: {len(y)} samples, {y.sum()} positive")
        
    return splits


def impute_with_model(model_name, checkpoint_path, split_data, batch_size=64, device='cuda'):
    """Usa un modello deep learning o baseline semplice per imputare i dati."""
    data = split_data['data']
    mask = split_data['mask']
    delta = split_data.get('delta', None)
    
    if model_name in ["mean", "locf", "linear_interp"]:
        from baselines_simple.simple import MeanImputer, LOCFImputer, LinearInterpImputer
        if model_name == "mean": imputer = MeanImputer()
        elif model_name == "locf": imputer = LOCFImputer()
        else: imputer = LinearInterpImputer()
        
        imputer.fit(data, mask) # LOCF e Linear non usano il fit in realtà
        imputed_data = imputer.impute(data, mask)
        
        # Ricostruisci il tensore combinando osservazioni vere e imputazioni
        final_imputed = (mask * data) + ((1 - mask) * imputed_data)
        return final_imputed

    elif any(model_name.startswith(base) for base in ["saits", "sssd", "brits", "mrnn", "gpvae", "timesfm"]):
        # Carica modello PyTorch Lightning
        base_name = [base for base in ["saits", "sssd", "brits", "mrnn", "gpvae", "timesfm"] if model_name.startswith(base)][0]
        if base_name == "saits":
            from models.saits.model import SAITSModule
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            # Heuristic: GraphDMSABlock uses output_proj with shape [1, 64], DMSABlock uses [D, 64]
            # Since D is usually 17 or 9 (not 1), we can distinguish.
            is_graph = "dmsa_block_1.output_proj.weight" in ckpt["state_dict"] and ckpt["state_dict"]["dmsa_block_1.output_proj.weight"].shape[0] == 1
            # Infer KGI from model name or checkpoint path string
            is_kgi = "kgi" in model_name.lower() or "kgi" in checkpoint_path.lower()
            model = SAITSModule.load_from_checkpoint(checkpoint_path, map_location=device, use_graph_layer=is_graph, use_kgi=is_kgi, strict=False, weights_only=False)
        elif base_name == "sssd":
            from models.sssd.model import SSSDModule
            model = SSSDModule.load_from_checkpoint(checkpoint_path, map_location=device, strict=False, weights_only=False)
            model.inference_samples = 1
        elif base_name == "brits":
            from models.brits.model import BRITSModule
            model = BRITSModule.load_from_checkpoint(checkpoint_path, map_location=device, strict=False, weights_only=False)
        elif base_name == "mrnn":
            from models.mrnn.model import MRNNModule
            model = MRNNModule.load_from_checkpoint(checkpoint_path, map_location=device, strict=False, weights_only=False)
        elif base_name == "gpvae":
            from models.gpvae.model import GPVAEModule
            model = GPVAEModule.load_from_checkpoint(checkpoint_path, map_location=device, strict=False, weights_only=False)
        elif base_name == "timesfm" or model_name == "timesfm_sapbert":
            from models.timesfm.model import TimesFMModule
            model = TimesFMModule.load_from_checkpoint(checkpoint_path, map_location=device, strict=False, weights_only=False)

        model = model.to(device)
        model.eval()

        # Dataloader
        class ImputeDataset(torch.utils.data.Dataset):
            def __init__(self, data, mask, delta):
                self.data = torch.from_numpy(data).float()
                self.mask = torch.from_numpy(mask).float()
                self.delta = torch.from_numpy(delta).float() if delta is not None else None
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                item = {
                    "data": self.data[idx],
                    "input_mask": self.mask[idx], # passiamo la maschera vera come input_mask (valori osservati)
                    "orig_mask": self.mask[idx],  # passiamo la maschera vera come orig_mask
                    "artificial_mask": torch.zeros_like(self.mask[idx]) # Nessuna maschera artificiale al test downstream
                }
                if self.delta is not None:
                    item["delta"] = self.delta[idx]
                return item

        loader = DataLoader(ImputeDataset(data, mask, delta), batch_size=batch_size, shuffle=False)
        trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1, logger=False)
        
        logger.info(f"Running inference for model {model_name}...")
        predictions = trainer.predict(model, dataloaders=loader)
        
        if isinstance(predictions[0], dict) and "imputed_3" in predictions[0]:
            predictions = torch.cat([p["imputed_3"] for p in predictions], dim=0).cpu().numpy()
        elif isinstance(predictions[0], dict) and "imputed" in predictions[0]:
            predictions = torch.cat([p["imputed"] for p in predictions], dim=0).cpu().numpy()
        else:
            predictions = torch.cat(predictions, dim=0).cpu().numpy()
        
        if predictions.shape != mask.shape:
             logger.warning(f"Shape mismatch: predictions {predictions.shape} vs mask {mask.shape}. Squeezing...")
             predictions = np.squeeze(predictions)
             if predictions.shape != mask.shape:
                  raise ValueError(f"CRITICAL: Shape mismatch after squeeze: {predictions.shape} vs {mask.shape}")

        final_imputed = (mask * data) + ((np.ones_like(mask) - mask) * predictions)
        del model, trainer
        import gc; gc.collect()
        torch.cuda.empty_cache()
        return final_imputed

    else:
        raise ValueError(f"Unknown model: {model_name}")


def extract_temporal_features(X):
    """Estrae feature temporali per la classificazione classica. X shape: (N, T, D)"""
    feats = []
    # Usiamo np.nanmin/nanmax per sicurezza, anche se dopo l'imputazione non dovrebbero esserci NaN
    X_safe = np.nan_to_num(X, nan=0.0)
    feats.append(np.min(X_safe, axis=1))
    feats.append(np.max(X_safe, axis=1))
    feats.append(np.mean(X_safe, axis=1))
    feats.append(X_safe[:, -1, :]) # l'ultimo valore disponibile
    return np.concatenate(feats, axis=1)


class GRUClassifier(pl.LightningModule):
    def __init__(self, input_dim, pos_weight, hidden_dim=64, num_layers=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.lr = lr

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return out.squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def evaluate_downstream(args):
    # 1. Labels
    if args.task == "sepsis":
        sepsis_cohort_path = Path(args.data_dir) / "sepsis_cohort.parquet"
        labels_df = generate_sepsis_outcome_labels(
            sepsis_cohort_path, pred_window_hours=args.pred_window_hours
        )
    else:
        labels_df = generate_decompensation_labels(
            args.icustays_path, args.admissions_path, 
            obs_window_hours=args.obs_window_hours, pred_window_hours=args.pred_window_hours
        )

    # 2. Data Alignment
    splits = load_and_align_data(Path(args.data_dir), labels_df)
    
    # 3. Imputation
    logger.info(f"Generating imputations using {args.model_name}...")
    X_imputed = {}
    for split in ['train', 'val', 'test']:
        X_imputed[split] = impute_with_model(args.model_name, args.checkpoint, splits[split], device=args.device)

    # =======================================================
    # APPROACH 1: LIGHTGBM on Temporal Features
    # =======================================================
    logger.info("--- Training LightGBM on Temporal Features ---")
    feat_train = extract_temporal_features(X_imputed['train'])
    feat_val = extract_temporal_features(X_imputed['val'])
    feat_test = extract_temporal_features(X_imputed['test'])

    lgb_train = lgb.Dataset(feat_train, label=splits['train']['y'])
    lgb_val = lgb.Dataset(feat_val, label=splits['val']['y'], reference=lgb_train)

    # Calcolo pos_weight dinamico
    pos_weight = (len(splits['train']['y']) - splits['train']['y'].sum()) / splits['train']['y'].sum()
    
    params = {
        'objective': 'binary',
        'metric': ['auc', 'average_precision'],
        'scale_pos_weight': pos_weight,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'num_threads': 4,
        'seed': 42,
        'verbose': -1
    }

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    y_pred_lgbm = gbm.predict(feat_test)
    auroc_lgbm = roc_auc_score(splits['test']['y'], y_pred_lgbm)
    auprc_lgbm = average_precision_score(splits['test']['y'], y_pred_lgbm)
    
    logger.info(f"[LightGBM] AUROC: {auroc_lgbm:.4f} | AUPRC: {auprc_lgbm:.4f}")

    # =======================================================
    # APPROACH 2: SOTA Deep Learning Classifier (GRU)
    # =======================================================
    logger.info("--- Training GRU on Fully Imputed Tensors ---")
    
    def create_dataloader(X, M, y, batch_size, shuffle=False):
        X_combined = np.concatenate([X, M], axis=-1)
        ds = TensorDataset(torch.from_numpy(X_combined).float(), torch.from_numpy(y).float())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    train_loader = create_dataloader(X_imputed['train'], splits['train']['mask'], splits['train']['y'], batch_size=128, shuffle=True)
    val_loader = create_dataloader(X_imputed['val'], splits['val']['mask'], splits['val']['y'], batch_size=256)
    test_loader = create_dataloader(X_imputed['test'], splits['test']['mask'], splits['test']['y'], batch_size=256)

    input_dim = X_imputed['train'].shape[2] * 2
    
    gru_model = GRUClassifier(input_dim=input_dim, pos_weight=pos_weight, hidden_dim=64, num_layers=2)
    
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
        enable_progress_bar=True,
        logger=False
    )

    trainer.fit(gru_model, train_loader, val_loader)
    
    # Valutazione
    gru_model.eval()
    preds_gru = []
    trues_gru = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(gru_model.device)
            logits = gru_model(x)
            probs = torch.sigmoid(logits)
            preds_gru.append(probs.cpu().numpy())
            trues_gru.append(y.numpy())

    preds_gru = np.concatenate(preds_gru)
    trues_gru = np.concatenate(trues_gru)
    
    auroc_gru = roc_auc_score(trues_gru, preds_gru)
    auprc_gru = average_precision_score(trues_gru, preds_gru)
    
    logger.info(f"[GRU Model] AUROC: {auroc_gru:.4f} | AUPRC: {auprc_gru:.4f}")
    
    # Save the results
    import datetime
    
    # Check if results folder exists
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result_dict = {
        "model": args.model_name,
        "setup": args.setup,
        "lgbm_auroc": float(auroc_lgbm),
        "lgbm_auprc": float(auprc_lgbm),
        "gru_auroc": float(auroc_gru),
        "gru_auprc": float(auprc_gru),
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    df_result = pd.DataFrame([result_dict])
    csv_out_path = results_dir / f"mortality_{args.model_name}_{args.setup}.csv"
    df_result.to_csv(csv_out_path, index=False)
    
    # Append to master benchmark
    master_out_path = results_dir / "mortality_master_benchmark.csv"
    if not master_out_path.exists():
        df_result.to_csv(master_out_path, index=False)
    else:
        df_result.to_csv(master_out_path, mode='a', header=False, index=False)
        
    logger.info(f"Results saved to {csv_out_path} and {master_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="decompensation", choices=["decompensation", "sepsis"], help="Task labels to generate")
    parser.add_argument("--model_name", type=str, required=True, help="saits, sssd, locf, mean, etc.")
    parser.add_argument("--setup", type=str, required=True, choices=["sota", "handpicked"], help="Which dataset setup is being evaluated")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (per deep models)")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path directory NPZ")
    parser.add_argument("--icustays_path", type=str, default="/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1/icu/icustays.csv.gz")
    parser.add_argument("--admissions_path", type=str, default="/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz")
    parser.add_argument("--obs_window_hours", type=int, default=48)
    parser.add_argument("--pred_window_hours", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128, help="Inferenza batch size")
    
    args = parser.parse_args()
    evaluate_downstream(args)
