import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error

from data.dataset import MIMICSepsisTaskDataset
from models.sepsis_lstm.model import LSTMModel
from models.sepsis_transformer.model import TimeSeriesTransformer

def extract_all_data(dataset):
    """Estrae l'intero dataset in numpy arrays X e y."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    X_list, y_list = [], []
    for batch in tqdm(loader, desc="Extracting data from Dataset"):
        # Concatenate data and mask to give the model awareness of missing values
        x = torch.cat([batch['data'], batch['input_mask']], dim=-1)
        X_list.append(x.numpy())
        y_list.append(batch['label'].numpy())
        
    return np.concatenate(X_list), np.concatenate(y_list).flatten()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['ihm', 'los', 'vr', 'ss'])
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'transformer'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use_kgi', action='store_true', help="Attiva l'iniezioni di conoscenza dinamica nel Transformer")
    args = parser.parse_args()
    
    # Load Datasets
    print(f"Loading {args.task.upper()} datasets...")
    train_data = MIMICSepsisTaskDataset("data/processed_sepsis/train.npz", task=args.task)
    val_data = MIMICSepsisTaskDataset("data/processed_sepsis/val.npz", task=args.task)
    test_data = MIMICSepsisTaskDataset("data/processed_sepsis/test.npz", task=args.task)
    
    print("Extracting Train...")
    X_train, y_train = extract_all_data(train_data)
    print("Extracting Val...")
    X_val, y_val = extract_all_data(val_data)
    print("Extracting Test...")
    X_test, y_test = extract_all_data(test_data)
    
    input_dim = X_train.shape[-1]
    task_type = 'regression' if args.task == 'los' else 'classification'
    
    # Initialize Model
    if args.model == 'lstm':
        model = LSTMModel(input_dim=input_dim, task_type=task_type)
    else:
        model = TimeSeriesTransformer(input_dim=input_dim, task_type=task_type, use_kgi=args.use_kgi)
        
    print(f"\n--- Training {args.model.upper()} on {args.task.upper()} ---")
    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape[1:]}")
    
    fit_kwargs = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr
    }
    if args.model == 'transformer':
        fit_kwargs['validation_data'] = (X_val, y_val)
        
    model.fit(X_train, y_train, **fit_kwargs)
    
    print("\n--- Evaluation on Test Set ---")
    preds = model.predict(X_test, batch_size=args.batch_size)
    
    if task_type == 'classification':
        auroc = roc_auc_score(y_test, preds)
        auprc = average_precision_score(y_test, preds)
        print(f"Test AUROC: {auroc:.4f}")
        print(f"Test AUPRC: {auprc:.4f}")
    else:
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"Test MAE:  {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()
