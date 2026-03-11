import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from data.dataset import configure_task

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerSepsisClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, nhead=8, dropout=0.1):
        super().__init__()
        # 1. Input projection to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 4. Output classification head (on last timestep)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x is [batch, seq_len, input_dim]
        # Project
        x = self.input_proj(x)
        # Add PE
        x = self.pos_encoder(x)
        # Transform
        x = self.transformer_encoder(x)
        # Last timestep classification (Paper strategy)
        x_last = x[:, -1, :]
        out = self.output_layer(x_last)
        return out.squeeze(-1)

class TransformerImputer(nn.Module):
    def __init__(self, d_feature, seq_len, d_model=64, n_heads=8, n_layers=2, dropout=0.1, use_kgi=False, kgi_embedding_file=None):
        super().__init__()
        self.d_feature = d_feature
        self.use_kgi = use_kgi
        self.input_proj = nn.Linear(d_feature * 2, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        if use_kgi:
            import os
            import pickle
            from models.saits.kgi_layer import DynamicKnowledgeInjector
            from models.sepsis_transformer.model import KGITransformerLayer
            
            self.shared_kgi = DynamicKnowledgeInjector(text_embed_dim=768, hidden_dim=d_model)
            with open(kgi_embedding_file, "rb") as f:
                self.medbert_dict = pickle.load(f)
            
            # Default ItemIDs (will be overridden by JointSepsisModule if subset active)
            self.kgi_itemids = list(range(d_feature)) 

            self.layers = nn.ModuleList([
                KGITransformerLayer(d_model, n_heads, d_model * 4, dropout, self.shared_kgi)
                for _ in range(n_layers)
            ])
        else:
            encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        self.reconstruction_head = nn.Linear(d_model, d_feature)

    def forward(self, batch):
        x = torch.cat([batch["data"], batch["input_mask"]], dim=-1)
        x = self.pos_encoder(self.input_proj(x))
        
        if self.use_kgi:
            surviving_mask = batch["input_mask"].bool()
            for layer in self.layers:
                x = layer(x, surviving_mask, self.medbert_dict, self.kgi_itemids)
        else:
            x = self.transformer_encoder(x)
            
        imputed = self.reconstruction_head(x)
        # Apply original mask
        imputed = batch["data"] * batch["input_mask"] + imputed * (1 - batch["input_mask"])
        return {"imputed_3": imputed}

class JointSepsisModule(pl.LightningModule):
    def __init__(
        self,
        imputator_name: str,
        imputator_kwargs: dict,
        d_feature: int = 55,
        task: str = 'ihm',
        alpha: float = 0.1,  # Weight for Imputation Loss
        beta: float = 1.0,   # Weight for Classification Loss
        pos_weight: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        obs_bins: int = 6,    # Number of bins used for classification (6 bins = 24h)
        feature_indices: list[int] | None = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.task = task.lower()
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.weight_decay = weight_decay
        self.obs_bins = obs_bins
        
        # Instantiate Imputer
        if "saits" in imputator_name:
            from models.saits.model import SAITSModule
            # Harmonize kwargs for SAITSModule
            self.imputer = SAITSModule(
                d_feature=d_feature,
                seq_len=imputator_kwargs.get("n_steps", 24),
                n_heads=imputator_kwargs.get("n_head", 8),
                **{k: v for k, v in imputator_kwargs.items() if k not in ["d_feature", "seq_len", "n_heads", "n_steps", "n_head"]}
            )
            # Override KGI mapping if subset active
            if feature_indices is not None and getattr(self.imputer, "use_kgi", False):
                original_full_itemids = self.imputer.kgi_itemids
                subset_itemids = [original_full_itemids[i] for i in feature_indices]
                self.imputer.kgi_itemids = subset_itemids
        elif "mrnn" in imputator_name:
            from models.mrnn.model import MRNNModule
            self.imputer = MRNNModule(**imputator_kwargs)
        elif "brits" in imputator_name:
            from models.brits.model import BRITSModule
            self.imputer = BRITSModule(**imputator_kwargs)
        elif "transformer" in imputator_name:
            self.imputer = TransformerImputer(
                d_feature=d_feature,
                seq_len=imputator_kwargs.get("n_steps", 24),
                use_kgi=imputator_kwargs.get("use_kgi", False),
                kgi_embedding_file=imputator_kwargs.get("kgi_embedding_file")
            )
            # Alignment for KGI if subset is active
            if feature_indices is not None and self.imputer.use_kgi:
                # Need the full list to slice
                kgi_itemids_full = [
                    220045, 220179, 225310, 220181, 220210, 220277, 223762, # Vitals
                    220074, 220059, 220060, 220061, 228368,                 # Hemodynamics
                    225664, 50813, 220615, 225690, 227457, 220546, 223835, 223830, # Labs 1
                    220224, 220235, 227442, 220645, 220602, 220635, 225625, 50804, # Labs 2
                    220587, 220644, 220650, 227456, 227429, 227444, 220228, 220545, # Labs 3
                    51279, 227465, 227466, 227467, 227443, 51006,                  # Labs 4
                    226755, 228096,                                                # Neuro
                    223834, 220339, 224686, 224687, 224697, 224695, 224696,        # Resp
                    9991, 9992, 9993, 9994 # Urine, Norepi, Fluid, Vent (Grounded)
                ]
                self.imputer.kgi_itemids = [kgi_itemids_full[i] for i in feature_indices]
        elif "sssd" in imputator_name:
            from models.sssd.model import SSSDModule
            self.imputer = SSSDModule(
                d_feature=d_feature,
                residual_layers=imputator_kwargs.get("residual_layers", 8),
                residual_channels=imputator_kwargs.get("residual_channels", 128),
                skip_channels=imputator_kwargs.get("skip_channels", 128),
                diffusion_embed_dim=imputator_kwargs.get("diffusion_embedding_dim", 128),
                s4_state_dim=imputator_kwargs.get("s4_state_dim", 128),
                s4_dropout=imputator_kwargs.get("s4_dropout", 0.2),
                T=imputator_kwargs.get("T", 200),
                beta_start=imputator_kwargs.get("beta_start", 1e-4),
                beta_end=imputator_kwargs.get("beta_end", 0.02),
                lr=lr,
                weight_decay=weight_decay,
                seq_len=imputator_kwargs.get("n_steps", 24),
                inference_steps=imputator_kwargs.get("inference_steps", 20),
                inference_samples=imputator_kwargs.get("inference_samples", 1),
                use_graph_prior=imputator_kwargs.get("use_graph_prior", False)
            )
        else:
            raise ValueError(f"Unknown imputer {imputator_name}")
            
        # Aligned Transformer Classifier (Huang et al. 2025)
        self.classifier = TransformerSepsisClassifier(input_dim=d_feature * 2) 
        
        # Loss and Metrics
        self.criterion_cls, self.val_metrics = configure_task(self.task, pos_weight)
        self.val_metrics = nn.ModuleDict(self.val_metrics)
        self.test_metrics = nn.ModuleDict({f"test_{k}": v.clone() for k, v in self.val_metrics.items()})

    def forward(self, batch):
        # 1. Imputation on the FULL sequence
        outputs = self.imputer(batch)
        
        if isinstance(outputs, dict):
            if "imputed_3" in outputs:
                imputed = outputs["imputed_3"]
            elif "imputed_data" in outputs:
                imputed = outputs["imputed_data"]
            else:
                imputed = outputs["reconstruction"]
        else:
            # SSSD case: returns the tensor directly
            imputed = outputs

        # 2. Slice for Classification (first 24h)
        imputed_slice = imputed[:, :self.obs_bins, :]
        mask_slice = batch["input_mask"][:, :self.obs_bins, :]
        
        # 3. Concatenate and Classify
        class_input = torch.cat([imputed_slice, mask_slice], dim=-1)
        logits = self.classifier(class_input)
        return logits, imputed

    def training_step(self, batch, batch_idx):
        # Multi-task training
        logits, imputed = self(batch)
        
        target = batch["label"].squeeze().float()
        loss_cls = self.criterion_cls(logits, target)
        
        # Self-supervised MAE on original missingness
        loss_imp = torch.mean(torch.abs(imputed - batch["data"]) * batch["input_mask"])

        total_loss = self.alpha * loss_imp + self.beta * loss_cls
        
        self.log("train/loss_cls", loss_cls, prog_bar=True)
        self.log("train/loss_imp", loss_imp)
        self.log("train/total_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        logits, _ = self(batch)
        target = batch["label"].squeeze().float()
        
        if self.task in ['ihm', 'vr', 'ss']:
            preds = torch.sigmoid(logits)
            targets = target.int()
        else:
            preds = logits
            targets = target
            
        for name, metric in self.val_metrics.items():
            metric(preds, targets)
            self.log(f"val/{name}", metric, on_epoch=True, prog_bar=True)
            
        self.log("val/loss", self.criterion_cls(logits, target))

    def test_step(self, batch, batch_idx):
        logits, _ = self(batch)
        target = batch["label"].squeeze().float()
        
        preds = torch.sigmoid(logits) if self.task != 'los' else logits
        targets = target.int() if self.task != 'los' else target
            
        for name, metric in self.test_metrics.items():
            metric(preds, targets)
            self.log(f"test/{name.replace('test_', '')}", metric, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
