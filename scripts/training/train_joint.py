"""
Joint Training Entrypoint

Hydra-configured training script for Joint Imputation + Classification models.
"""

import logging
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader, Dataset

from models.joint.model import JointTrainingModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Re-use label generation from evaluate_downstream
def generate_decompensation_labels(icustays_path, admissions_path, obs_window_hours=48, pred_window_hours=24):
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
    return final_labels

class JointDataset(Dataset):
    def __init__(self, npz_path, labels_df):
        loaded = np.load(npz_path)
        self.data = loaded["data"].astype(np.float32)
        self.orig_mask = loaded["orig_mask"].astype(np.float32)
        if "delta" in loaded:
            self.delta = loaded["delta"].astype(np.float32)
        else:
            self.delta = np.zeros_like(self.data)
            
        stay_ids = loaded["stay_ids"]
        
        df_stays = pd.DataFrame({'stay_id': stay_ids, 'idx': np.arange(len(stay_ids))})
        merged = pd.merge(df_stays, labels_df, on='stay_id', how='inner')
        
        idx = merged['idx'].values
        self.y = merged['label_decompensation'].values.astype(np.float32)
        
        self.data = self.data[idx]
        self.orig_mask = self.orig_mask[idx]
        self.delta = self.delta[idx]

    def __len__(self):
         return len(self.data)

    def __getitem__(self, idx):
         return {
             "data": torch.from_numpy(self.data[idx]),
             "orig_mask": torch.from_numpy(self.orig_mask[idx]),
             "input_mask": torch.from_numpy(self.orig_mask[idx]), # full mask as input
             "artificial_mask": torch.zeros_like(torch.from_numpy(self.orig_mask[idx])), # No artificial masking during joint
             "delta": torch.from_numpy(self.delta[idx]),
             "target": torch.from_numpy(self.data[idx]),
             "label": torch.tensor([self.y[idx]], dtype=torch.float32)
         }

class JointDataModule(pl.LightningDataModule):
    def __init__(self, processed_dir: str, labels_df: pd.DataFrame, batch_size=64, num_workers=4):
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.labels_df = labels_df
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = JointDataset(self.processed_dir / "train.npz", self.labels_df)
            self.val_dataset = JointDataset(self.processed_dir / "val.npz", self.labels_df)
        if stage == "test" or stage is None:
            self.test_dataset = JointDataset(self.processed_dir / "test.npz", self.labels_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

def get_imputator_kwargs(cfg, d_feature, seq_len):
    model_name = cfg.model.name
    if model_name == "saits":
        return {
            "d_feature": d_feature,
            "d_model": cfg.model.d_model,
            "d_inner": cfg.model.d_inner,
            "n_heads": cfg.model.n_heads,
            "d_k": cfg.model.d_k,
            "d_v": cfg.model.d_v,
            "n_layers": cfg.model.n_layers,
            "n_dmsa_blocks": cfg.model.n_dmsa_blocks,
            "dropout": cfg.model.dropout,
            "alpha": cfg.model.alpha,
            "seq_len": seq_len,
            "graph_loss_weight": cfg.model.get("graph_loss_weight", 1.0),
            "use_prior_init": cfg.model.get("use_prior_init", True),
            "warmup_epochs": cfg.model.get("warmup_epochs", 0),
            "dag_loss_weight": cfg.model.get("dag_loss_weight", 0.0),
            "use_graph_layer": cfg.model.get("use_graph_layer", True),
            "parallel_attention": cfg.model.get("parallel_attention", True),
            "embedding_type": cfg.model.get("embedding_type", "vanilla"),
            "use_kgi": cfg.model.get("use_kgi", False),
            "kgi_embedding_file": cfg.model.get("kgi_embedding_file", "medbert_relation_embeddings_generic.pkl")
        }
    elif model_name in ["brits", "mrnn"]:
        return {
            "d_feature": d_feature,
            "seq_len": seq_len,
            "rnn_hidden_size": cfg.model.get("rnn_hidden_size", 64),
            "use_kgi": cfg.model.get("use_kgi", False),
            "kgi_embedding_file": cfg.model.get("kgi_embedding_file", "medbert_relation_embeddings_generic.pkl")
        }
    elif "timesfm" in model_name:
        return {
            "d_feature": d_feature,
            "seq_len": seq_len,
            "use_kgi": cfg.model.get("use_kgi", False),
            "kgi_embedding_file": cfg.model.get("kgi_embedding_file", "medbert_relation_embeddings_generic.pkl"),
            "use_graph_layer": cfg.model.get("use_graph_layer", False),
            "graph_loss_weight": cfg.model.get("graph_loss_weight", 0.0),
            "embedding_type": cfg.model.get("embedding_type", "vanilla")
        }
    else:
        raise ValueError(f"Unknown imputator for joint training: {model_name}")

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)

    # 1. Labels
    logger.info("Generating labels...")
    labels_df = generate_decompensation_labels(
        "/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1/icu/icustays.csv.gz", 
        "/home/guido/Data/indus_data/physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz"
    )

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    datamodule = JointDataModule(
        processed_dir=data_cfg["processed_dir"],
        labels_df=labels_df,
        batch_size=data_cfg.get("batch_size", 64),
        num_workers=data_cfg.get("num_workers", 4)
    )
    datamodule.setup("fit")
    
    d_feature = len(cfg.data.feature_names)
    seq_len = 48 # standard

    pos_weight = (len(datamodule.train_dataset.y) - datamodule.train_dataset.y.sum()) / datamodule.train_dataset.y.sum()

    kwargs = get_imputator_kwargs(cfg, d_feature, seq_len)
    
    # 2. Model
    alpha_joint = cfg.model.get("alpha_joint", 0.1)
    beta_joint = cfg.model.get("beta_joint", 1.0)
    
    model = JointTrainingModule(
        imputator_name=cfg.model.name,
        imputator_kwargs=kwargs,
        d_feature=d_feature,
        alpha=alpha_joint, 
        beta=beta_joint,
        pos_weight=pos_weight,
        lr=cfg.model.optimizer.lr,
        weight_decay=cfg.model.optimizer.weight_decay
    )

    # 3. Callbacks
    emb_type = cfg.model.get("embedding_type", "vanilla")
    use_kgi = "_KGI" if cfg.model.get("use_kgi", False) else "_Vanilla"
    base_checkpoint_dir = f"outputs/checkpoints/joint_{cfg.model.name}"
    # Add alpha to path for distinct experiments
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"{emb_type}{use_kgi}_a{alpha_joint}")

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:02d}-{val/auprc:.4f}",
            monitor="val/auprc",
            mode="max",
            save_top_k=1,
            save_last=True
        ),
        EarlyStopping(
            monitor="val/auprc",
            patience=10,
            mode="max"
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]

    from pytorch_lightning.loggers import TensorBoardLogger
    logger_tb = TensorBoardLogger(
        save_dir="outputs/logs",
        name=f"joint_{cfg.model.name}{use_kgi}",
    )

    # 4. Train
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    
    # We want less epochs for joint usually or keep from config
    max_epochs = min(trainer_cfg.get("max_epochs", 30), 40)
    trainer_cfg["max_epochs"] = max_epochs
    
    trainer = pl.Trainer(
        **trainer_cfg,
        callbacks=callbacks,
        logger=logger_tb,
        deterministic=True
    )

    logger.info("Starting JOINT training...")
    trainer.fit(model, datamodule=datamodule)
    logger.info("Joint training complete.")

if __name__ == "__main__":
    main()
