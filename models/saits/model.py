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
        kgi_mode: str = "dki", # 'dki' (input), 'dgi' (layer v1), 'dgi_mask' (layer v2)
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
        self.kgi_mode = kgi_mode
        
        # Mask-Aware logic (for DGI v2)
        self.mask_aware = (kgi_mode == "dgi_mask")
        
        # GSL Logic: Use GraphDMSABlock if graph prior is requested
        self.use_graph_prior = (embedding_type != "vanilla")
        
        # Layers
        self.input_proj = nn.Linear(d_feature * 2, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)
        
        block_kwargs = {
            "n_layers": n_layers, "n_heads": n_heads, "d_model": d_model, 
            "d_k": d_k, "d_v": d_v, "d_inner": d_inner, "dropout": dropout, 
            "d_feature": d_feature, "mask_aware": self.mask_aware
        }
        
        # Architecture Selection
        if "dgi" in self.kgi_mode:
            from models.saits.layers import GatedSemanticBlock
            block_class = GatedSemanticBlock # NO GNN, direct injection
        elif self.use_graph_prior:
            block_class = GraphDMSABlock
        else:
            block_class = DMSABlock
            
        self.dmsa_block_1 = block_class(**block_kwargs)
        self.dmsa_block_2 = block_class(**block_kwargs)
        self.combining_weight = nn.Linear(d_feature, d_feature)

        # KGI
        self.use_kgi = use_kgi
        if self.use_kgi:
            import os
            import pickle
            from models.saits.kgi_layer import KGIFusionLayer, DynamicKnowledgeInjector
            
            kgi_file = kwargs.get("kgi_embedding_file", "medbert_relation_embeddings_sepsis.pkl")
            
            # Robust Path Logic
            base_path = os.getcwd() 
            if kgi_file.startswith("data/"):
                kgi_path = os.path.join(base_path, kgi_file)
            else:
                kgi_path = os.path.join(base_path, "data/embeddings", kgi_file)
            
            with open(kgi_path, 'rb') as f:
                self.medbert_dict = pickle.load(f)
            
            vocab_path = os.path.join(base_path, "data/embeddings/mimic_vocab_mapped.csv")
            vocab = pd.read_csv(vocab_path)
            self.kgi_itemids = vocab['itemid'].tolist()
            
            # Input-level Fusion (used in DKI mode)
            self.kgi_fusion = KGIFusionLayer(hidden_dim=d_model)
            
            # Layer-level Injector (used in DGI mode)
            if self.kgi_mode == "dgi":
                self.kgi_injector = DynamicKnowledgeInjector(text_embed_dim=768, hidden_dim=d_model)

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

        # 1. Project input
        # Feature-wise stack: (B, T, D, 2)
        x_feat = torch.stack([data * input_mask, input_mask], dim=-1)
        x_proj = self.pos_encoding(self.input_proj(torch.cat([data * input_mask, input_mask], dim=-1)))
        
        # 2. Knowledge Injection context
        feature_embeddings = None
        if self.use_kgi:
            if self.kgi_mode == "dki":
                # Apply fusion once at the input level
                x_proj = self.kgi_fusion(x_proj, surviving_mask, self.medbert_dict, self.kgi_itemids)
            elif self.kgi_mode == "dgi":
                # Generate embeddings context using current temporal projections
                # feature_embeddings: (B, T, D, H) based on MedBERT/SapBERT logic
                feature_embeddings = self.kgi_injector(x_proj, surviving_mask, self.medbert_dict, self.kgi_itemids)

        # 3. Blocks execution
        if "dgi" in self.kgi_mode:
            # DGI Path (Gated Semantic Injection - supports both v1 and v2/mask)
            h1, imp1, _ = self.dmsa_block_1(x_feat, feature_embeddings=feature_embeddings)
            x_replaced = data * input_mask + imp1 * (1 - input_mask)
            x_feat_2 = torch.stack([x_replaced, input_mask], dim=-1)
            h2, imp2, _ = self.dmsa_block_2(x_feat_2, feature_embeddings=feature_embeddings)
        elif self.use_graph_prior:
            # GNN Path
            h1, imp1, _ = self.dmsa_block_1(x_feat)
            x_replaced = data * input_mask + imp1 * (1 - input_mask)
            h2, imp2, _ = self.dmsa_block_2(torch.stack([x_replaced, input_mask], dim=-1))
        else:
            # Vanilla SAITS Path
            h1, imp1, _ = self.dmsa_block_1(x_proj)
            x_proj_2 = self.pos_encoding(self.input_proj(torch.cat([data * input_mask + imp1 * (1 - input_mask), input_mask], dim=-1)))
            h2, imp2, _ = self.dmsa_block_2(x_proj_2)

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
