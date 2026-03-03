import torch
import torch.nn as nn
import pytorch_lightning as pl
from pypots.imputation import MRNN as PyPOTSMRNN
from pypots.nn.modules.mrnn.layers import MrnnFcnRegression
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
import os
import pickle

from metrics.imputation import mae
from models.saits.kgi_layer import DynamicKnowledgeInjector

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
        # x: (B, T_slice, Hidden)
        x_mean = torch.mean(x, dim=1)
        x_max, _ = torch.max(x, dim=1)
        x_combined = torch.cat([x_mean, x_max], dim=-1)
        return self.net(x_combined)

class KGI_MrnnFcnRegression(MrnnFcnRegression):
    def __init__(self, feature_num, text_embed_dim=768):
        super().__init__(feature_num)
        self.kgi_injector = DynamicKnowledgeInjector(text_embed_dim=text_embed_dim, hidden_dim=feature_num)
        self.kgi_final_linear = nn.Linear(feature_num * 2, feature_num)
        self.medbert_dict = None
        self.kgi_itemids = None

    def forward(self, x, missing_mask, target, surviving_mask_t=None):
        h_t = torch.sigmoid(F.linear(x, self.U * self.m) + F.linear(target, self.V1 * self.m) + F.linear(missing_mask, self.V2) + self.beta)
        if surviving_mask_t is None or self.medbert_dict is None:
            kgi_context = torch.zeros_like(h_t)
        else:
            mask_3d = surviving_mask_t.unsqueeze(1) if surviving_mask_t.dim() == 2 else surviving_mask_t
            query_3d = h_t.unsqueeze(1) if h_t.dim() == 2 else h_t
            kgi_context = self.kgi_injector(query_hidden=query_3d, surviving_mask=mask_3d, precomputed_embeddings=self.medbert_dict, variable_indices=self.kgi_itemids)
            if h_t.dim() == 2: kgi_context = kgi_context.squeeze(1)
        return torch.sigmoid(self.kgi_final_linear(torch.cat([h_t, kgi_context], dim=-1)))

class MRNNModule(pl.LightningModule):
    def __init__(
        self,
        d_feature: int,
        seq_len: int,
        rnn_hidden_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0,
        task_type: str = "binary",
        pos_weight: float = 5.5,
        obs_steps: int = 2,
        imp_weight: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.use_kgi = kwargs.get("use_kgi", False)
        self.obs_steps = obs_steps
        self.imp_weight = imp_weight
        self.task_type = task_type
        
        self.pypots_model = PyPOTSMRNN(n_steps=seq_len, n_features=d_feature, rnn_hidden_size=rnn_hidden_size, epochs=1)
        self.model = self.pypots_model.model
        
        if self.use_kgi:
            self.model.backbone.fcn_regression = KGI_MrnnFcnRegression(d_feature, text_embed_dim=768)
            kgi_file = kwargs.get("kgi_embedding_file", "medbert_relation_embeddings_sepsis.pkl")
            with open(os.path.expanduser(f"~/Code/charite/baselines/data/embeddings/{kgi_file}"), 'rb') as f:
                self.medbert_dict = pickle.load(f)
            vocab = pd.read_csv(os.path.expanduser("~/Code/charite/baselines/data/embeddings/mimic_vocab_mapped.csv"))
            self.kgi_itemids = vocab['itemid'].tolist()[:d_feature]

        self.classifier = DownstreamClassifier(input_dim=rnn_hidden_size * 2, hidden_dim=128, task_type=task_type)
        if task_type == "regression":
            self.task_loss_fn = nn.MSELoss()
        else:
            pw = torch.tensor([pos_weight]) if pos_weight else None
            self.task_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
            
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()
        self.test_auprc = BinaryAveragePrecision()
        self.lr = lr

    def forward(self, batch: dict) -> dict:
        # Prepare inputs for the underlying MRNN model
        inputs = {
            "forward": {
                "X": batch["data"],
                "missing_mask": 1 - batch["input_mask"],
                "deltas": batch["delta"]
            },
            "backward": {
                "X": torch.flip(batch["data"], [1]),
                "missing_mask": 1 - torch.flip(batch["input_mask"], [1]),
                "deltas": torch.flip(batch["delta"], [1])
            },
            "X_ori": batch["data"],
            "indicating_mask": batch["input_mask"]
        }
        return self._forward_model(inputs, batch)

    def _forward_model(self, inputs, batch):
        X, M = inputs["forward"]["X"], inputs["forward"]["missing_mask"]
        self.model.backbone.fcn_regression.medbert_dict = self.medbert_dict if self.use_kgi else None
        self.model.backbone.fcn_regression.kgi_itemids = self.kgi_itemids if self.use_kgi else None

        feature_collector, hidden_collector = [], []
        for f in range(self.model.backbone.n_features):
            feat_est, hid_f, _ = self.model.backbone.gene_hidden_states(inputs, f)
            feature_collector.append(feat_est); hidden_collector.append(hid_f)

        RNN_est = torch.concat(feature_collector, dim=2)
        RNN_imp = (1 - M) * X + M * RNN_est
        combined_hidden = torch.stack(hidden_collector, dim=-1).mean(dim=-1)
        
        surviving_mask = batch["input_mask"].bool() & ~(batch.get("artificial_mask", torch.zeros_like(batch["input_mask"])).bool())
        
        if self.use_kgi:
            FCN_est = self.model.backbone.fcn_regression(X, M, RNN_imp, surviving_mask)
        else:
            FCN_est = self.model.backbone.fcn_regression(X, M, RNN_imp)
        
        # Slicing for classification
        logits = self.classifier(combined_hidden[:, :self.obs_steps, :])
        return {"imputed_3": (1 - M) * X + M * FCN_est, "logits": logits}

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss_imp = (torch.abs(out["imputed_3"] - batch["target"]) * batch["artificial_mask"]).sum() / batch["artificial_mask"].sum().clamp(min=1)
        loss_pred = self.task_loss_fn(out["logits"].squeeze(-1), batch["labels"].float())
        total_loss = loss_pred + self.imp_weight * loss_imp
        self.log("train/total_loss", total_loss, prog_bar=True); self.log("train/pred_loss", loss_pred, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        val_mae = mae(out["imputed_3"], batch["target"], batch["artificial_mask"])
        probs = torch.sigmoid(out["logits"].squeeze(-1))
        self.val_auroc.update(probs, batch["labels"].long()); self.val_auprc.update(probs, batch["labels"].long())
        self.log("val/mae", val_mae, prog_bar=True); self.log("val/auroc", self.val_auroc, on_epoch=True, prog_bar=True)
        self.log("val/loss", 0.0) # Placeholder

    def test_step(self, batch, batch_idx):
        out = self(batch)
        probs = torch.sigmoid(out["logits"].squeeze(-1))
        self.test_auroc.update(probs, batch["labels"].long())
        self.test_auprc.update(probs, batch["labels"].long())
        self.log("test/auroc", self.test_auroc, on_epoch=True)
        self.log("test/auprc", self.test_auprc, on_epoch=True)
        self.log("test/mae", mae(out["imputed_3"], batch["target"], batch["artificial_mask"]))
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
