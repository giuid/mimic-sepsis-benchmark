import torch
import torch.nn as nn
import pytorch_lightning as pl
from pypots.imputation import BRITS as PyPOTSBRITS
from pypots.nn.modules.brits.backbone import BackboneRITS, BackboneBRITS
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
import os
import pickle
from typing import Tuple

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

class KGI_BackboneRITS(BackboneRITS):
    def __init__(self, n_steps: int, n_features: int, rnn_hidden_size: int, training_loss, text_embed_dim=768):
        super().__init__(n_steps, n_features, rnn_hidden_size, training_loss)
        self.step_kgi = DynamicKnowledgeInjector(text_embed_dim=text_embed_dim, hidden_dim=rnn_hidden_size)
        self.rnn_cell = nn.LSTMCell((self.n_features * 2) + rnn_hidden_size, self.rnn_hidden_size)
        self.medbert_dict, self.kgi_itemids = None, None

    def forward(self, inputs: dict, direction: str, surviving_mask: torch.Tensor = None) -> Tuple:
        X, M, D = inputs[direction]["X"], inputs[direction]["missing_mask"], inputs[direction]["deltas"]
        device = X.device
        hidden_states = torch.zeros((X.size()[0], self.rnn_hidden_size), device=device)
        cell_states = torch.zeros((X.size()[0], self.rnn_hidden_size), device=device)
        estimations, all_hidden = [], []
        reconstruction_loss = torch.tensor(0.0, device=device)

        for t in range(self.n_steps):
            x, m, d = X[:, t, :], M[:, t, :], D[:, t, :]
            gamma_h, gamma_x = self.temp_decay_h(d), self.temp_decay_x(d)
            hidden_states = hidden_states * gamma_h
            x_h = self.hist_reg(hidden_states)
            reconstruction_loss += self.training_loss(x_h, x, m)
            x_c = m * x + (1 - m) * x_h
            z_h = self.feat_reg(x_c)
            reconstruction_loss += self.training_loss(z_h, x, m)
            alpha = torch.sigmoid(self.combining_weight(torch.cat([gamma_x, m], dim=1)))
            c_h = alpha * z_h + (1 - alpha) * x_h
            reconstruction_loss += self.training_loss(c_h, x, m)
            c_c = m * x + (1 - m) * c_h
            estimations.append(c_h.unsqueeze(dim=1))

            if surviving_mask is None or self.medbert_dict is None:
                kgi_t = torch.zeros((X.size()[0], self.step_kgi.hidden_dim), device=device)
            else:
                kgi_t = self.step_kgi(query_hidden=hidden_states.unsqueeze(1), surviving_mask=surviving_mask[:, t, :].unsqueeze(1), precomputed_embeddings=self.medbert_dict, variable_indices=self.kgi_itemids).squeeze(1)

            hidden_states, cell_states = self.rnn_cell(torch.cat([c_c, m, kgi_t], dim=1), (hidden_states, cell_states))
            all_hidden.append(hidden_states.unsqueeze(1))

        return (1 - M) * X + M * torch.cat(estimations, dim=1), torch.cat(estimations, dim=1), torch.cat(all_hidden, dim=1), reconstruction_loss / (self.n_steps * 3)

class KGI_BackboneBRITS(BackboneBRITS):
    def __init__(self, n_steps: int, n_features: int, rnn_hidden_size: int, training_loss, text_embed_dim=768):
        nn.Module.__init__(self)
        self.n_steps, self.n_features, self.rnn_hidden_size = n_steps, n_features, rnn_hidden_size
        self.rits_f = KGI_BackboneRITS(n_steps, n_features, rnn_hidden_size, training_loss, text_embed_dim)
        self.rits_b = KGI_BackboneRITS(n_steps, n_features, rnn_hidden_size, training_loss, text_embed_dim)
        
    def forward(self, inputs: dict, surviving_mask_f: torch.Tensor = None, surviving_mask_b: torch.Tensor = None) -> Tuple:
        (f_imp, f_rec, f_hid, f_loss) = self.rits_f(inputs, "forward", surviving_mask_f)
        (b_imp, b_rec, b_hid, b_loss) = self._reverse(self.rits_b(inputs, "backward", surviving_mask_b))
        return (f_imp + b_imp) / 2, f_rec, b_rec, f_hid, b_hid, self._get_consistency_loss(f_imp, b_imp), f_loss + b_loss

class BRITSModule(pl.LightningModule):
    def __init__(self, d_feature: int, seq_len: int, rnn_hidden_size: int = 64, lr: float = 1e-3, weight_decay: float = 0, task_type: str = "binary", pos_weight: float = 5.5, obs_steps: int = 2, imp_weight: float = 0.1, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.use_kgi, self.obs_steps, self.imp_weight, self.task_type = kwargs.get("use_kgi", False), obs_steps, imp_weight, task_type
        self.pypots_model = PyPOTSBRITS(n_steps=seq_len, n_features=d_feature, rnn_hidden_size=rnn_hidden_size, epochs=1)
        print(f"BRITS: Initializing Unified Backbone (KGI support: {self.use_kgi})...")
        self.pypots_model.model.model = KGI_BackboneBRITS(seq_len, d_feature, rnn_hidden_size, self.pypots_model.model.model.rits_f.training_loss, text_embed_dim=768)
        self.model = self.pypots_model.model
        if self.use_kgi:
            with open(os.path.expanduser(f"~/Code/charite/baselines/data/embeddings/{kwargs.get('kgi_embedding_file', 'medbert_relation_embeddings_sepsis.pkl')}"), 'rb') as f: self.medbert_dict = pickle.load(f)
            self.kgi_itemids = pd.read_csv(os.path.expanduser("~/Code/charite/baselines/data/embeddings/mimic_vocab_mapped.csv"))['itemid'].tolist()[:d_feature]
        else: self.medbert_dict, self.kgi_itemids = None, None
        self.classifier = DownstreamClassifier(input_dim=rnn_hidden_size * 4, hidden_dim=128, task_type=task_type)
        self.task_loss_fn = nn.MSELoss() if task_type == "regression" else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]) if pos_weight else None)
        self.val_auroc = BinaryAUROC(); self.val_auprc = BinaryAveragePrecision(); 
        self.test_auroc = BinaryAUROC(); self.test_auprc = BinaryAveragePrecision();
        self.lr = lr

    def forward(self, batch: dict) -> dict:
        # Prepare inputs for the underlying BRITS model
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
        orig_mask = batch["input_mask"]
        if self.use_kgi:
            for rits in [self.model.model.rits_f, self.model.model.rits_b]: rits.medbert_dict, rits.kgi_itemids = self.medbert_dict, self.kgi_itemids
            out = self.model.model(inputs, orig_mask.bool(), torch.flip(orig_mask.bool(), [1]))
        else: out = self.model.model(inputs)
        hidden = torch.cat([out[3], out[4]], dim=-1)
        return {"imputed_3": out[0], "loss": out[5] + out[6], "logits": self.classifier(hidden[:, :self.obs_steps, :])}

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss_pred = self.task_loss_fn(out["logits"].squeeze(-1), batch["labels"].float())
        total_loss = loss_pred + self.imp_weight * out["loss"]
        self.log("train/total_loss", total_loss, prog_bar=True); self.log("train/pred_loss", loss_pred, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        val_mae = mae(out["imputed_3"], batch["target"], batch["artificial_mask"])
        probs = torch.sigmoid(out["logits"].squeeze(-1))
        self.val_auroc.update(probs, batch["labels"].long()); self.val_auprc.update(probs, batch["labels"].long())
        self.log("val/mae", val_mae, prog_bar=True); self.log("val/auroc", self.val_auroc, on_epoch=True, prog_bar=True); self.log("val/loss", 0.0)

    def test_step(self, batch, batch_idx):
        out = self(batch)
        probs = torch.sigmoid(out["logits"].squeeze(-1))
        self.test_auroc.update(probs, batch["labels"].long())
        self.test_auprc.update(probs, batch["labels"].long())
        self.log("test/auroc", self.test_auroc, on_epoch=True)
        self.log("test/auprc", self.test_auprc, on_epoch=True)
        self.log("test/mae", mae(out["imputed_3"], batch["target"], batch["artificial_mask"]))
        return out

    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=self.lr)
