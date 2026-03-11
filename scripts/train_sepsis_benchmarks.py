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
    parser.add_argument('--use_kgi', action='store_true', help="Attiva l'iniezioni di conoscenza nel Transformer")
    parser.add_argument('--kgi_mode', type=str, default='dgi', choices=['dki', 'dgi', 'dgi_mask'], help="Modalità di iniezione: dki (input), dgi (layer-wise v1) o dgi_mask (layer-wise v2)")
    parser.add_argument('--kgi_alpha', type=float, default=1.0, help="Initial value for KGI alpha")
    parser.add_argument('--kgi_alpha_fixed', action='store_true', help="Freeze KGI alpha (non-trainable)")
    parser.add_argument('--feature_subset', type=str, default='full', choices=['full', 'no_treatments', 'core', 'emergency'], help="Subset of features to use")
    parser.add_argument('--data_dir', type=str, default="data/processed_sepsis_full", help="Directory containing .npz files")
    parser.add_argument('--kgi_embedding', type=str, default="data/embeddings/medbert_relation_embeddings_sepsis_full.pkl")
    parser.add_argument('--save_dir', type=str, default="checkpoints/sepsis", help="Directory to save the best model")
    parser.add_argument('--exclude_treatments', action='store_true', help="Drop the last 4 feature variables (treatments/interventions)")
    args = parser.parse_args()
    
    # Load Datasets
    print(f"Loading {args.task.upper()} datasets from {args.data_dir} (Subset: {args.feature_subset})...")
    train_data = MIMICSepsisTaskDataset(os.path.join(args.data_dir, "train.npz"), task=args.task, feature_subset=args.feature_subset)
    val_data = MIMICSepsisTaskDataset(os.path.join(args.data_dir, "val.npz"), task=args.task, feature_subset=args.feature_subset)
    test_data = MIMICSepsisTaskDataset(os.path.join(args.data_dir, "test.npz"), task=args.task, feature_subset=args.feature_subset)
    
    # Get global feature itemids for KGI mapping
    feature_indices = train_data.feature_indices
    
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
        model = TimeSeriesTransformer(
            input_dim=input_dim, 
            task_type=task_type, 
            use_kgi=args.use_kgi,
            kgi_mode=args.kgi_mode,
            kgi_alpha_value=args.kgi_alpha,
            kgi_alpha_trainable=not args.kgi_alpha_fixed,
            kgi_embedding_file=args.kgi_embedding
        )
        
        # Override the default kgi_itemids_full using the selected indices
        if args.use_kgi:
            # We need to map the selected subset indices back to their original itemids
            # The model's kgi_itemids_full is already aligned with the 55-feature YAML.
            original_full_itemids = model.kgi_itemids_full
            subset_itemids = [original_full_itemids[i] for i in feature_indices]
            model.kgi_itemids_full = subset_itemids
        
    print(f"\n--- Training {args.model.upper()} on {args.task.upper()} ---")
    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape[1:]}")
    
    fit_kwargs = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr
    }
    
    if task_type == 'classification':
        num_pos = np.sum(y_train)
        num_neg = len(y_train) - num_pos
        pos_weight = float(num_neg / max(num_pos, 1.0))
        fit_kwargs['pos_weight'] = pos_weight
        print(f"Calculated pos_weight: {pos_weight:.4f} (Pos: {num_pos}, Neg: {num_neg})")

    # Pass validation data for early stopping to ALL models
    fit_kwargs['validation_data'] = (X_val, y_val)
    
    # Generate save path (Hierarchical & Unique)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # In TimeSeriesTransformer, use_kgi can be 'dki' or 'dgi'
    if args.use_kgi:
        inj_base = args.kgi_mode # 'dki' or 'dgi'
        inj_type = f"{inj_base}_adaptive" if not args.kgi_alpha_fixed else f"{inj_base}_fixed"
    else:
        inj_type = "vanilla"
        
    task_dir = os.path.join(args.save_dir, args.task, args.feature_subset)
    os.makedirs(task_dir, exist_ok=True)
    
    filename = f"{args.model}_{args.task}_{args.feature_subset}_{inj_type}_{timestamp}.pt"
    save_path = os.path.join(task_dir, filename)
    
    fit_kwargs['save_path'] = save_path
    print(f"Model will be saved to: {save_path}")
        
    model.fit(X_train, y_train, **fit_kwargs)
    
    print("\n--- Evaluation on Test Set ---")
    preds = model.predict(X_test, batch_size=args.batch_size)
    
    final_results = {
        "task": args.task,
        "subset": args.feature_subset,
        "variant": "dki_fixed" if args.use_kgi and args.kgi_alpha_fixed else ("dki_adaptive" if args.use_kgi else "vanilla"),
        "alpha_init": args.kgi_alpha,
        "alpha_final": model.kgi_alpha.item() if (args.use_kgi and hasattr(model, 'kgi_alpha')) else 0.0,
        "kgi_mag": getattr(model, 'last_kgi_rel_mag', 0.0) if args.use_kgi else 0.0
    }

    if task_type == 'classification':
        auroc = roc_auc_score(y_test, preds)
        auprc = average_precision_score(y_test, preds)
        print(f"Test AUROC: {auroc:.4f}")
        print(f"Test AUPRC: {auprc:.4f}")
        final_results.update({"auroc": auroc, "auprc": auprc})
    else:
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"Test MAE:  {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        final_results.update({"mae": mae, "rmse": rmse})

    # Append to master ablation CSV
    import pandas as pd
    from datetime import datetime
    final_results["timestamp"] = datetime.now().isoformat()
    res_path = "results/sepsis_ablation_study.csv"
    os.makedirs("results", exist_ok=True)
    df_new = pd.DataFrame([final_results])
    if os.path.exists(res_path):
        df_old = pd.read_csv(res_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(res_path, index=False)
    else:
        df_new.to_csv(res_path, index=False)
    print(f"Results appended to {res_path}")

if __name__ == "__main__":
    main()
