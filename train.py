"""
Training Entrypoint

Hydra-configured training script for SAITS and SSSD models.

Usage:
    # Train SAITS with random masking (30%)
    python train.py model=saits masking=random masking.p=0.3

    # Train SSSD with block masking
    python train.py model=sssd masking=block

    # Quick dev run
    python train.py model=saits trainer.fast_dev_run=true

    # Override any config
    python train.py model=saits model.lr=5e-4 trainer.max_epochs=50
"""

import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from data.dataset import MIMICDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_model(cfg: DictConfig, d_feature: int, seq_len: int) -> pl.LightningModule:
    """
    Build model from config.

    Args:
        cfg: full Hydra config
        d_feature: number of features D
        seq_len: sequence length T

    Returns:
        LightningModule instance
    """
    model_name = cfg.model.name

    if model_name == "saits":
        from models.saits.model import SAITSModule

        return SAITSModule(
            d_feature=d_feature,
            d_model=cfg.model.d_model,
            d_inner=cfg.model.d_inner,
            n_heads=cfg.model.n_heads,
            d_k=cfg.model.d_k,
            d_v=cfg.model.d_v,
            n_layers=cfg.model.n_layers,
            n_dmsa_blocks=cfg.model.n_dmsa_blocks,
            dropout=cfg.model.dropout,
            alpha=cfg.model.alpha,
            lr=cfg.model.optimizer.lr,
            weight_decay=cfg.model.optimizer.weight_decay,
            seq_len=seq_len,
            graph_loss_weight=cfg.model.get("graph_loss_weight", 1.0),
            use_prior_init=cfg.model.get("use_prior_init", True),
            warmup_epochs=cfg.model.get("warmup_epochs", 0),
            dag_loss_weight=cfg.model.get("dag_loss_weight", 0.0),
            use_graph_layer=cfg.model.get("use_graph_layer", True),
            parallel_attention=cfg.model.get("parallel_attention", True),
            embedding_type=cfg.model.get("embedding_type", "vanilla"),
            use_kgi=cfg.model.get("use_kgi", False),
            kgi_embedding_file=cfg.model.get("kgi_embedding_file", "medbert_relation_embeddings_generic.pkl"),
        )

    elif model_name == "sssd":
        from models.sssd.model import SSSDModule

        return SSSDModule(
            d_feature=d_feature,
            residual_layers=cfg.model.residual_layers,
            residual_channels=cfg.model.residual_channels,
            skip_channels=cfg.model.skip_channels,
            diffusion_embed_dim=cfg.model.diffusion_embedding_dim,
            s4_state_dim=cfg.model.s4.state_dim,
            s4_dropout=cfg.model.s4.dropout,
            T=cfg.model.diffusion.T,
            beta_start=cfg.model.diffusion.beta_start,
            beta_end=cfg.model.diffusion.beta_end,
            lr=cfg.model.optimizer.lr,
            weight_decay=cfg.model.optimizer.weight_decay,
            seq_len=seq_len,
            inference_steps=cfg.model.diffusion.get("inference_steps", 1000),
            inference_samples=cfg.model.get("inference_samples", 10),
            use_graph_prior=cfg.model.get("use_graph_prior", True),
        )

    elif model_name == "brits":
        from models.brits.model import BRITSModule
        return BRITSModule(
            d_feature=len(cfg.data.feature_names),
            seq_len=seq_len,
            rnn_hidden_size=cfg.model.get("rnn_hidden_size", 64),
            lr=cfg.model.optimizer.lr,
            weight_decay=cfg.model.optimizer.weight_decay,
        )

    elif model_name == "mrnn":
        from models.mrnn.model import MRNNModule
        return MRNNModule(
            d_feature=len(cfg.data.feature_names),
            seq_len=seq_len,
            rnn_hidden_size=cfg.model.get("rnn_hidden_size", 64),
            lr=cfg.model.optimizer.lr,
            weight_decay=cfg.model.optimizer.weight_decay,
        )

    elif model_name == "gpvae":
        from models.gpvae.model import GPVAEModule
        return GPVAEModule(
            d_feature=len(cfg.data.feature_names),
            seq_len=seq_len,
            latent_size=cfg.model.get("latent_size", 64),
            encoder_sizes=cfg.model.get("encoder_sizes", [128, 128]),
            decoder_sizes=cfg.model.get("decoder_sizes", [128, 128]),
            lr=cfg.model.optimizer.lr,
            weight_decay=cfg.model.optimizer.weight_decay,
        )

    elif model_name.startswith("timesfm"):
        from models.timesfm.model import TimesFMModule
        return TimesFMModule(
            d_feature=d_feature,
            seq_len=seq_len,
            lr=cfg.model.optimizer.lr,
            weight_decay=cfg.model.optimizer.weight_decay,
            model_id=cfg.model.model_id,
            embedding_type=cfg.model.get("embedding_type", "vanilla"),
            use_graph_layer=cfg.model.get("use_graph_layer", False),
            graph_loss_weight=cfg.model.get("graph_loss_weight", 0.0),
            kgi_embedding_file=cfg.model.get("kgi_embedding_file", "medbert_relation_embeddings_generic.pkl"),
        )

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: saits, sssd, brits, mrnn, gpvae")


def build_callbacks(cfg: DictConfig) -> list:
    """Build Lightning callbacks."""
    callbacks = []

    # Checkpoint: save best model by val/loss
    base_checkpoint_dir = cfg.get("checkpoint_dir", f"outputs/checkpoints/{cfg.model.name}")
    
    # DISAMBIGUATION: Extract unique identifiers so concurrent runs do not overlap
    emb_type = cfg.model.get("embedding_type", "default")
    use_kgi = "_KGI" if cfg.model.get("use_kgi", False) else ""
    
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"{emb_type}{use_kgi}")

    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            verbose=True,
        )
    )

    # Early stopping
    callbacks.append(
        EarlyStopping(
            monitor="val/loss",
            patience=20,
            mode="min",
            verbose=True,
        )
    )

    # LR monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    return callbacks


def build_logger(cfg: DictConfig):
    """Build training logger (WandB or TensorBoard)."""
    log_cfg = cfg.get("logging", {})
    logger_type = log_cfg.get("logger", "tensorboard")

    if logger_type == "wandb":
        # Construct descriptive name: [Model]_[Emb]_[Prior?]_[Dataset]_[Masking]
        model_name = cfg.model.name.upper()
        dataset = "SOTA" if "sota" in cfg.data.processed_dir.lower() else "HANDPICKED"
        masking = cfg.masking.name.capitalize()
        
        if cfg.model.name == "saits":
            emb = cfg.model.get("embedding_type", "vanilla").capitalize()
            prior = "_Prior" if cfg.model.get("use_prior_init", False) else ""
            kgi = "_KGI" if cfg.model.get("use_kgi", False) else ""
            name = f"SAITS_{emb}{prior}{kgi}_{dataset}_{masking}"
        else:
            name = f"{model_name}_{dataset}_{masking}"

        try:
            from pytorch_lightning.loggers import WandbLogger
            return WandbLogger(
                project=log_cfg.get("project", "mimic-iv-imputation"),
                name=name,
                save_dir=log_cfg.get("log_dir", "outputs/logs"),
            )
        except ImportError:
            logger.warning("WandB not available, falling back to TensorBoard")

    from pytorch_lightning.loggers import TensorBoardLogger
    return TensorBoardLogger(
        save_dir=log_cfg.get("log_dir", "outputs/logs"),
        name=f"{cfg.model.name}_{cfg.masking.name}",
    )


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main training entrypoint."""

    # Print config
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 60)

    # Set seed
    pl.seed_everything(cfg.seed, workers=True)

    # Build data module
    masking_cfg = OmegaConf.to_container(cfg.masking, resolve=True)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

    datamodule = MIMICDataModule(
        processed_dir=data_cfg["processed_dir"],
        masking_cfg=masking_cfg,
        batch_size=data_cfg.get("batch_size", 64),
        num_workers=data_cfg.get("num_workers", 4),
        eval_seed=masking_cfg.get("eval_seed", 42),
    )

    # Setup to get dimensions
    datamodule.setup("fit")
    d_feature = datamodule.feature_dim
    seq_len = datamodule.seq_len

    logger.info("Dataset: D=%d, T=%d, N_train=%d, N_val=%d",
                d_feature, seq_len,
                len(datamodule.train_dataset),
                len(datamodule.val_dataset))

    # Build model
    model = build_model(cfg, d_feature=d_feature, seq_len=seq_len)
    logger.info("Model: %s (%d parameters)",
                cfg.model.name,
                sum(p.numel() for p in model.parameters()))

    # Build trainer
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    callbacks = build_callbacks(cfg)
    train_logger = build_logger(cfg)

    trainer = pl.Trainer(
        **trainer_cfg,
        callbacks=callbacks,
        logger=train_logger,
        deterministic=True,
    )

    # Save initial metadata
    save_metadata(cfg, model, datamodule, best_ckpt="Running...")

    # Train
    logger.info("Starting training...")
    ckpt_path = cfg.get("checkpoint")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    logger.info("Training complete!")

    # Log best checkpoint path
    best_ckpt = callbacks[0].best_model_path
    logger.info("Best checkpoint: %s", best_ckpt)
    logger.info("Best val/loss: %.6f", callbacks[0].best_model_score)

    # Save detailed metadata
    save_metadata(cfg, model, datamodule, best_ckpt)


def save_metadata(cfg: DictConfig, model: pl.LightningModule, datamodule: MIMICDataModule, best_ckpt: str):
    """Save detailed experiment metadata to a text file."""
    import datetime
    import os
    
    # Always save to cfg.output_dir (where checkpoints go)
    output_dir = cfg.get("output_dir", os.getcwd())
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    meta_path = os.path.join(output_dir, "experiment_metadata.txt")
    
    dataset_name = "SOTA" if "sota" in cfg.data.processed_dir.lower() else "HANDPICKED"
    
    with open(meta_path, "w") as f:
        f.write(f"Experiment Metadata\n")
        f.write(f"===================\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Model: {cfg.model.name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Masking: {cfg.masking.name} (p={cfg.masking.get('p', 'N/A')})\n")
        f.write(f"\nData Stats:\n")
        f.write(f"  Features (D): {datamodule.feature_dim}\n")
        f.write(f"  Seq Len (T): {datamodule.seq_len}\n")
        f.write(f"  Train Samples: {len(datamodule.train_dataset)}\n")
        f.write(f"  Val Samples: {len(datamodule.val_dataset)}\n")
        
        f.write(f"\nModel Config:\n")
        f.write(f"  d_model: {cfg.model.get('d_model', 'N/A')}\n")
        f.write(f"  n_layers: {cfg.model.get('n_layers', 'N/A')}\n")
        
        if cfg.model.name == "saits":
             f.write(f"  Graph Loss Weight: {cfg.model.get('graph_loss_weight', 0.0)}\n")
             f.write(f"  DAG Loss Weight: {cfg.model.get('dag_loss_weight', 0.0)}\n")
             f.write(f"  Use Prior Init: {cfg.model.get('use_prior_init', False)}\n")
             # Log graph info if available
             if hasattr(model, "dmsa_blocks"):
                 # Check first block
                 block = model.dmsa_blocks[0]
                 if hasattr(block, "get_graph_structure"):
                      f.write(f"  Graph Architecture: Parallel GraphDMSABlock\n")
        
        f.write(f"\nTraining Results:\n")
        f.write(f"  Best Checkpoint: {best_ckpt}\n")
        
        f.write(f"\nFull Config Dump:\n")
        f.write(OmegaConf.to_yaml(cfg))
        
    logger.info("Saved experiment metadata to %s", meta_path)


if __name__ == "__main__":
    main()
