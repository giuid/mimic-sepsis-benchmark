import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data.dataset import MIMICSepsisTaskDataset
from models.joint.sepsis_model import JointSepsisModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["saits", "mrnn", "brits"])
    parser.add_argument("--task", type=str, required=True, choices=["ihm", "los", "vr", "ss"])
    parser.add_argument("--kgi", action="store_true", help="Usa varianti KGI invece di Vanilla")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"Loading data for task: {args.task.upper()}")
    train_ds = MIMICSepsisTaskDataset("data/processed_sepsis/train.npz", task=args.task)
    val_ds = MIMICSepsisTaskDataset("data/processed_sepsis/val.npz", task=args.task)
    test_ds = MIMICSepsisTaskDataset("data/processed_sepsis/test.npz", task=args.task)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Sequence length is statically 2 bins for all tasks (6h truncated or 6h lookback)
    n_steps = 2
    n_features = 22
    
    # 2. Configure Imputer Model parameters
    imputator_kwargs = {
        "seq_len": n_steps,
        "d_feature": n_features,
        "rnn_hidden_size": 64 if args.model in ["mrnn", "brits"] else None,
        "n_layers": 2,
        "d_model": 64,
        "d_inner": 128,
        "n_heads": 4,
        "d_k": 16,
        "d_v": 16,
        "dropout": 0.1,
    }
    
    # Filter out kwargs explicitly handled by each class
    if args.model in ["mrnn", "brits"]:
        imputator_kwargs = {k: v for k, v in imputator_kwargs.items() if k in ["seq_len", "d_feature", "rnn_hidden_size"]}
    else: # saits
        imputator_kwargs = {k: v for k, v in imputator_kwargs.items() if k not in ["rnn_hidden_size"]}
        
    # KGI Injection
    if args.kgi:
        imputator_kwargs["use_kgi"] = True
        imputator_kwargs["kgi_embedding_file"] = "medbert_relation_embeddings_sepsis.pkl"

    # 3. Instantiate Joint Model
    model = JointSepsisModule(
        imputator_name=args.model,
        imputator_kwargs=imputator_kwargs,
        d_feature=n_features,
        task=args.task,
        lr=args.lr
    )
    
    # 4. Logger and Callbacks
    model_type = f"{args.model}_kgi" if args.kgi else f"{args.model}_vanilla"
    exp_name = f"sepsis_{args.task}_{model_type}"
    logger = TensorBoardLogger("tb_logs_sepsis", name=exp_name)
    
    monitor_metric = "val/rmse" if args.task == 'los' else "val/auprc"
    monitor_mode = "min" if args.task == 'los' else "max"
    
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode=monitor_mode,
        dirpath=f"checkpoints/sepsis/{exp_name}",
        filename="best-{epoch:02d}",
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor=monitor_metric, patience=5, mode=monitor_mode)
    
    # 5. Train
    print(f"Training {model_type.upper()} on {args.task.upper()}...")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[args.gpu],
        logger=logger,
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=10
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    # 6. Test
    print(f"\nEvaluating {model_type.upper()} on Test Set...")
    trainer.test(model, test_loader, ckpt_path="best")

if __name__ == "__main__":
    main()
