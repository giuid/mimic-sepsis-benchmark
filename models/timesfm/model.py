import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import numpy as np
import pickle
import pandas as pd
from typing import Optional
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
from models.timesfm.layers import GraphFeatureInteraction
from models.saits.kgi_layer import KGIFusionLayer
from metrics.imputation import mae_torch, mae, rmse, mre, r2_score, correlation_error

class TimesFMModule(pl.LightningModule):
    """
    Multivariate TimesFM for clinical time-series imputation.
    """
    def __init__(
        self,
        d_feature: int,
        seq_len: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        embedding_type: str = "vanilla", # "vanilla", "sapbert"
        use_graph_layer: bool = False,
        graph_loss_weight: float = 0.0,
        use_kgi: bool = False,
        kgi_embedding_file: str = "medbert_relation_embeddings_generic.pkl",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.d_feature = d_feature
        self.seq_len = seq_len
        self.lr = lr
        self.weight_decay = weight_decay
        self.graph_loss_weight = graph_loss_weight
        self.use_kgi = use_kgi
        self.kgi_embedding_file = kgi_embedding_file
        
        # 1. Initialize TimesFM Architecture
        self.timesfm_base = TimesFM_2p5_200M_torch_module()
        
        # 2. Load Pretrained Weights (Internal Load logic)
        self._load_pretrained_weights(model_id)
        
        # 3. Multivariate Interaction Layer
        self.use_graph_layer = use_graph_layer
        if self.use_graph_layer:
            self.graph_layer = GraphFeatureInteraction(
                d_feature=d_feature,
                d_model=self.timesfm_base.md, # 1280
                embedding_type=embedding_type,
                use_prior_init=True
            )

        # 4. KGI Layer
        if self.use_kgi:
            print(f"TimesFM: Initializing Knowledge-Guided Imputation (KGI) Layer [{self.kgi_embedding_file}]...")
            self.kgi_fusion = KGIFusionLayer(hidden_dim=self.timesfm_base.md, text_embed_dim=768)
            
            # Load embeddings
            embed_path = os.path.join("data/embeddings", self.kgi_embedding_file)
            vocab_path = "data/embeddings/mimic_vocab_mapped.csv"
            
            if os.path.exists(embed_path):
                with open(embed_path, 'rb') as f:
                    self.medbert_dict = pickle.load(f)
            else:
                print(f"WARNING: KGI embedding file {embed_path} not found.")
                self.medbert_dict = {}
                
            if os.path.exists(vocab_path):
                vocab = pd.read_csv(vocab_path)
                self.kgi_itemids = vocab['itemid'].tolist()[:d_feature]
            else:
                print(f"WARNING: Vocabulary file {vocab_path} not found.")
                self.kgi_itemids = []
        
        # 5. Final Combination / Imputation Head
        # Custom Imputation Head: (d_model) -> (patch_len)
        # This replaces the 128-step forecasting head for better reconstruction
        self.imputation_head = nn.Linear(self.timesfm_base.md, 32) # patch_len is fixed to 32 in TimesFM 2.5
        
        self.combining_weight = nn.Linear(d_feature, d_feature, bias=False)

    def _load_pretrained_weights(self, model_id):
        weights_dir = os.path.expanduser("~/timesfm_25_weights")
        ckpt_path = os.path.join(weights_dir, "torch_model_correctly_mapped.ckpt")
        
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            self.timesfm_base.load_state_dict(sd, strict=False)
            print(f"TimesFM: Loaded pre-mapped weights from {ckpt_path}")
        else:
            print("TimesFM: Pre-mapped weights not found. You might need to run the mapping script.")

    def forward(self, batch):
        data = batch["data"]           # (B, T, D)
        input_mask = batch["input_mask"]  # (B, T, D)
        B, T, D = data.shape
        
        # 1. Pad to multiple of patch_len (32)
        patch_len = self.timesfm_base.p
        pad_len = (patch_len - (T % patch_len)) % patch_len
        if pad_len > 0:
            # Pad at the end
            x = torch.cat([data, torch.zeros(B, pad_len, D, device=self.device)], dim=1)
            m = torch.cat([input_mask, torch.zeros(B, pad_len, D, device=self.device)], dim=1)
        else:
            x = data
            m = input_mask
            
        T_padded = x.shape[1]
        N_patches = T_padded // patch_len
            
        # 2. Reshape and Patch: (B, T_p, D) -> (B*D, N_patches, patch_len)
        x = x.transpose(1, 2) # (B, D, T_p)
        x = x.reshape(B * D, N_patches, patch_len)
        
        m = m.transpose(1, 2) # (B, D, T_p)
        m = m.reshape(B * D, N_patches, patch_len)
        
        # 3. TimesFM Forward
        # TimesFM_2p5_200M_torch_module.forward expects (inputs, masks)
        # where inputs: (Batch, N_patches, patch_len), masks: (Batch, N_patches, patch_len)
        # (Actually masks should be boolean, 1=masked? No, tokenizer cats them.)
        # In timesfm_2p5_torch.py: torch.cat([inputs, masks.to(inputs.dtype)], dim=-1)
        # So masks should be 1.0 for padding/masking? 
        # In decode(): normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
        # So patched_masks=True means "masked".
        m_bool = (m == 0) # 1 in input_mask means observed, so 0 means masked
        
        features, _ = self.timesfm_base(x, m_bool)
        
        # input_emb = features[0] # (B*D, N_patches, d_model)
        output_emb = features[1] # (B*D, N_patches, d_model)
        
        # 4. Reshape back to multivariate: (B, D, N_patches, d_model)
        out_emb_mv = output_emb.reshape(B, D, N_patches, -1)
        
        # 4.5 KGI Injection (Textual Knowledge)
        if self.use_kgi:
            # Create a patch-level mask: (B, N_patches, D)
            # m shape is (B*D, N_patches, patch_len) at this point
            # Reshape back to (B, D, N_patches, patch_len) then aggregate
            m_4d = m.reshape(B, D, N_patches, patch_len)
            m_patch = (m_4d.sum(dim=-1) > 0).float() # (B, D, N_patches)
            m_patch = m_patch.transpose(1, 2) # (B, N_patches, D)
            
            # Global latent state for KGI (average across features)
            q_kgi = out_emb_mv.mean(dim=1) # (B, N_patches, d_model)
            
            # Fuse with Medical Knowledge
            fused_q = self.kgi_fusion(q_kgi, m_patch, self.medbert_dict, self.kgi_itemids)
            
            # Add fused context back to all features (broadcast residual)
            # fused_q: (B, N_patches, d_model) -> out_emb_mv: (B, D, N_patches, d_model)
            out_emb_mv = out_emb_mv + fused_q.unsqueeze(1)
            
        # 5. Cross-feature Interaction (Graph)
        if self.use_graph_layer:
            out_emb_mv = self.graph_layer(out_emb_mv)
            
        # 6. Final Imputation Projection
        # Use our custom head to project hidden states back to patch values (same patch)
        imputed_patches = self.imputation_head(out_emb_mv) # (B, D, N_patches, patch_len)
        
        # 7. Map patches back to sequence (No shifting)
        # imputed_patches: (B, D, N_patches, patch_len) -> (B, D, T_padded)
        imputed_seq = imputed_patches.reshape(B, D, -1)
        
        # 8. Unpad and reshape back: (B, D, T_padded) -> (B, T, D)
        imputed_3 = imputed_seq[:, :, :T].transpose(1, 2)
        
        # 9. Learned refinement across features (applied to potential imputed values BEFORE masked update)
        imputed_3 = self.combining_weight(imputed_3)
        
        # 10. Masked update (Keep observed values - original input_mask, not potentially modified one)
        orig_input_mask = batch.get("orig_mask", batch["input_mask"]) if not hasattr(batch, 'get') else batch.get("orig_mask", batch["input_mask"])
        imputed_3 = data * input_mask + imputed_3 * (1 - input_mask)
        
        return {
            "imputed_3": imputed_3
        }

    def _compute_loss(self, batch, outputs):
        target = batch["target"]
        artificial_mask = batch["artificial_mask"]
        # Use orig_mask if available (in case input_mask was modified during joint training masking)
        orig_input_mask = batch.get("orig_mask", batch["input_mask"])
        
        # MIT loss: only on artificially masked positions
        loss_mit = mae_torch(outputs["imputed_3"], target, artificial_mask)
        # ORT loss: on naturally observed positions (not modified by joint training)
        loss_ort = mae_torch(outputs["imputed_3"], target, orig_input_mask)
        
        loss = 0.8 * loss_mit + 0.2 * loss_ort
        
        loss_graph = torch.tensor(0.0, device=self.device)
        if self.use_graph_layer and self.graph_loss_weight > 0:
            A = self.graph_layer.get_adj()
            # Sparsity regularization
            loss_graph = torch.norm(A, p=1)
            loss += self.graph_loss_weight * loss_graph
            
        return {
            "loss": loss,
            "loss_mit": loss_mit.detach(),
            "loss_ort": loss_ort.detach(),
            "loss_graph": loss_graph.detach() if self.use_graph_layer else 0.0
        }

    def training_step(self, batch, batch_idx):
        # Proactively apply artificial masking if not present (for joint training signal)
        if "artificial_mask" in batch and batch["artificial_mask"].sum() == 0:
            # Create random mask on-the-fly to give imputer a target
            data = batch["data"]
            mask = torch.rand_like(data) < 0.2
            batch["artificial_mask"] = mask.float() * batch["input_mask"] # only mask observed
            batch["input_mask"] = batch["input_mask"] * (1 - batch["artificial_mask"])

        outputs = self(batch)
        losses = self._compute_loss(batch, outputs)
        self.log("train/loss", losses["loss"], prog_bar=True)
        self.log("train/loss_mit", losses["loss_mit"])
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        losses = self._compute_loss(batch, outputs)
        
        imputed = outputs["imputed_3"]
        target = batch["target"]
        mask = batch["artificial_mask"]
        
        val_mae = mae(imputed, target, mask)
        self.log("val/loss", losses["loss"], prog_bar=True)
        self.log("val/mae", val_mae, prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        imputed = outputs["imputed_3"]
        target = batch["target"]
        mask = batch["artificial_mask"]
        
        self.log("test/mae", mae(imputed, target, mask), sync_dist=True)
        self.log("test/rmse", rmse(imputed, target, mask), sync_dist=True)
        self.log("test/corr_err", correlation_error(imputed, target), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}
        }
