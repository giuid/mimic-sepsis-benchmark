"""
SAITS: Multi-task Polimorfico (Huang et al. 2025 Sepsis Benchmark Optimized)

Optimized for:
- Static Task: In-Hospital Mortality (IHM) using first 6-8h.
- Dynamic Balancing: pos_weight for imbalanced sepsis classes.
- Clinical Priority: Reduced imputation weight (0.1) vs Prediction weight (1.0).
"""

import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from models.saits.layers import DMSABlock, PositionalEncoding, GraphDMSABlock
from metrics.imputation import mae
import numpy as np

class DownstreamClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2, task_type="binary"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.task_type = task_type

    def forward(self, x):
        # Global pooling (mean + max) over the specified observation window
        x_mean = torch.mean(x, dim=1)
        x_max, _ = torch.max(x, dim=1)
        x_combined = torch.cat([x_mean, x_max], dim=-1)
        logits = self.net(x_combined)
        return logits

class SAITSModule(pl.LightningModule):
    def __init__(
        self,
        d_feature: int,
        seq_len: int,
        d_model: int,
        d_inner: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        n_layers: int,
        dropout: float,
        n_dmsa_blocks: int = 2,
        alpha: float = 0.8,
        lr: float = 1e-3,
        weight_decay: float = 0,
        embedding_type: str = "vanilla",
        use_kgi: bool = False,
        task_type: str = "binary",
        pos_weight: float = 5.5,
        obs_steps: int = 2, # First 8h (2 bins of 4h) for prediction
        imp_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.d_feature = d_feature
        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay
        self.obs_steps = obs_steps
        self.imp_weight = imp_weight
        self.task_type = task_type
        
        # GSL Logic
        self.use_graph_prior = (embedding_type != "vanilla")
        
        # Layers
        self.input_proj = nn.Linear(d_feature * 2, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)
        
        block_kwargs = {"n_layers": n_layers, "n_heads": n_heads, "d_model": d_model, "d_k": d_k, "d_v": d_v, "d_inner": d_inner, "dropout": dropout, "d_feature": d_feature}
        block_class = GraphDMSABlock if self.use_graph_prior else DMSABlock
        self.dmsa_block_1 = block_class(**block_kwargs)
        self.dmsa_block_2 = block_class(**block_kwargs)
        self.combining_weight = nn.Linear(d_feature, d_feature)

        # KGI
        self.use_kgi = use_kgi
        if self.use_kgi:
            from models.saits.kgi_layer import KGIFusionLayer
            kgi_file = kwargs.get("kgi_embedding_file", "medbert_relation_embeddings_sepsis.pkl")
            with open(os.path.expanduser(f"~/Code/charite/baselines/data/embeddings/{kgi_file}"), 'rb') as f:
                self.medbert_dict = pickle.load(f)
            vocab = pd.read_csv(os.path.expanduser("~/Code/charite/baselines/data/embeddings/mimic_vocab_mapped.csv"))
            self.kgi_itemids = vocab['itemid'].tolist()[:d_feature]
            self.kgi_fusion = KGIFusionLayer(hidden_dim=d_model)

        # Flexible Classification/Regression Head
        self.classifier = DownstreamClassifier(input_dim=d_model * 2, hidden_dim=d_inner, task_type=task_type)
        
        if task_type == "regression":
            self.task_loss_fn = nn.MSELoss()
        else:
            pw = torch.tensor([pos_weight]) if pos_weight else None
            self.task_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()

    def forward(self, batch: dict) -> dict:
        data, input_mask = batch["data"], batch["input_mask"]
        surviving_mask = input_mask.bool() & (~batch.get("artificial_mask", torch.zeros_like(input_mask)).bool())

        x = self.pos_encoding(self.input_proj(torch.cat([data * input_mask, input_mask], dim=-1)))
        if self.use_kgi:
            x = self.kgi_fusion(x, surviving_mask, self.medbert_dict, self.kgi_itemids)

        if self.use_graph_prior:
            # GNN path
            h1, imp1, _ = self.dmsa_block_1(torch.stack([data * input_mask, input_mask], dim=-1))
            x_replaced = data * input_mask + imp1 * (1 - input_mask)
            h2, imp2, _ = self.dmsa_block_2(torch.stack([x_replaced, input_mask], dim=-1))
        else:
            # Vanilla path
            h1, imp1, _ = self.dmsa_block_1(x)
            x2 = self.pos_encoding(self.input_proj(torch.cat([data * input_mask + imp1 * (1 - input_mask), input_mask], dim=-1)))
            h2, imp2, _ = self.dmsa_block_2(x2)

        imp3 = self.combining_weight((imp1 + imp2) / 2.0)
        imp3 = data * input_mask + imp3 * (1 - input_mask)

        # SLICING: Use only the first 'obs_steps' for prediction
        logits = self.classifier(h2[:, :self.obs_steps, :])

        return {"imputed_3": imp3, "imp1": imp1, "imp2": imp2, "logits": logits}

    def _compute_imp_loss(self, batch, outputs):
        target, mask = batch["target"], batch["artificial_mask"]
        loss_mit = (mae(outputs["imp1"], target, mask) + mae(outputs["imp2"], target, mask)) / 2.0
        loss_ort = mae(outputs["imputed_3"], target, batch["orig_mask"])
        return self.alpha * loss_mit + (1 - self.alpha) * loss_ort

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        imp_loss = self._compute_imp_loss(batch, outputs)
        
        labels = batch["labels"].float()
        pred_loss = self.task_loss_fn(outputs["logits"].squeeze(-1), labels)
        
        # Combined Loss: Prediction (1.0) + Imputation (0.1)
        total_loss = pred_loss + self.imp_weight * imp_loss

        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/pred_loss", pred_loss, prog_bar=True)
        self.log("train/imp_loss", imp_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        imp_loss = self._compute_imp_loss(batch, outputs)
        
        labels = batch["labels"].float()
        probs = torch.sigmoid(outputs["logits"].squeeze(-1))
        
        self.val_auroc.update(probs, labels.long())
        self.val_auprc.update(probs, labels.long())

        self.log("val/mae", mae(outputs["imputed_3"], batch["target"], batch["artificial_mask"]), prog_bar=True)
        self.log("val/auroc", self.val_auroc, on_epoch=True, prog_bar=True)
        self.log("val/loss", imp_loss) # Log imputation loss for early stopping if needed

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        labels = batch["labels"].float()
        probs = torch.sigmoid(outputs["logits"].squeeze(-1))
        
        self.test_auroc.update(probs, labels.long())
        self.test_auprc.update(probs, labels.long())

        self.log("test/auroc", self.test_auroc, on_epoch=True)
        self.log("test/auprc", self.test_auprc, on_epoch=True)
        self.log("test/mae", mae(outputs["imputed_3"], batch["target"], batch["artificial_mask"]))
        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
