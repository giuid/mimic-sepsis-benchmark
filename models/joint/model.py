import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

class JointGRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return out.squeeze(1)

class JointTrainingModule(pl.LightningModule):
    def __init__(
        self,
        imputator_name: str,
        imputator_kwargs: dict,
        d_feature: int = 17,
        alpha: float = 0.1,
        beta: float = 1.0,
        pos_weight: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        
        # Instantiate Imputer
        if "saits" in imputator_name:
            from models.saits.model import SAITSModule
            self.imputer = SAITSModule(**imputator_kwargs)
        elif "mrnn" in imputator_name:
            from models.mrnn.model import MRNNModule
            self.imputer = MRNNModule(**imputator_kwargs)
        elif "brits" in imputator_name:
            from models.brits.model import BRITSModule
            self.imputer = BRITSModule(**imputator_kwargs)
        elif "timesfm" in imputator_name:
            from models.timesfm.model import TimesFMModule
            self.imputer = TimesFMModule(**imputator_kwargs)
        else:
            raise ValueError(f"Unknown imputer {imputator_name}")
            
        self.classifier = JointGRUClassifier(input_dim=d_feature * 2) # X and Mask
        self.criterion_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]))
        
        # Caches
        self.val_preds = []
        self.val_targets = []
        self._sync_device()

    def _sync_device(self):
        if self.criterion_cls.pos_weight.device != self.device:
             self.criterion_cls.pos_weight = self.criterion_cls.pos_weight.to(self.device)

    def get_imputer_outputs_and_loss(self, batch, is_training: bool = False):
        if "timesfm" in self.hparams.imputator_name:
            # TimesFM branch
            # Proactively apply artificial masking ONLY DURING TRAINING to give imputer a signal
            if is_training and "artificial_mask" in batch and batch["artificial_mask"].sum() == 0:
                # Store the original mask for the loss computation
                batch["orig_mask"] = batch["input_mask"].clone()
                # Create random mask on-the-fly to give imputer a target
                r_mask = (torch.rand_like(batch["data"]) < 0.2)
                batch["artificial_mask"] = r_mask.float() * batch["input_mask"] # only mask observed
                batch["input_mask"] = batch["input_mask"] * (1 - batch["artificial_mask"])

            outputs = self.imputer(batch)
            losses = self.imputer._compute_loss(batch, outputs)
            imputed = outputs["imputed_3"]
            imp_loss = losses["loss"]
            self.log("train/imp_loss_mit", losses["loss_mit"])
            return imputed, imp_loss
            
        elif hasattr(self.imputer, '_compute_loss'):
            # SAITS branch
            outputs = self.imputer(batch)
            losses = self.imputer._compute_loss(batch, outputs)
            imputed = outputs["imputed_3"]
            
            # Combine all SAITS sub-losses
            imp_loss = losses["loss_mit"]
            if "loss_ort" in losses:
                imp_loss = imp_loss * self.imputer.alpha + losses["loss_ort"] * (1 - self.imputer.alpha)
            if "loss_graph" in losses and losses["loss_graph"]:
                imp_loss = imp_loss + losses["graph_weight"] * losses["loss_graph"]
            if "loss_dag" in losses and losses["loss_dag"]:
                imp_loss = imp_loss + losses["graph_weight"] * self.imputer.dag_loss_weight * losses["loss_dag"]
                
             # Log internal SAITS metrics
            self.log("train/imp_loss_mit", losses["loss_mit"])
            if "loss_ort" in losses: self.log("train/imp_loss_ort", losses["loss_ort"])
            return imputed, imp_loss
            
        elif hasattr(self.imputer, 'pypots_model'):
            # PyPOTS models (MRNN, BRITS)
            inputs = self.imputer._assemble_inputs(batch["data"], batch["input_mask"], batch.get("delta"))
            
            if not self.imputer.use_kgi:
                 out = self.imputer.model(inputs, calc_criterion=True)
                 if isinstance(out, dict):
                     imputed = out.get("imputed_data", out.get("reconstruction", out.get("FCN_estimation")))
                     imp_loss = out.get("loss", out.get("reconstruction_loss", 0) + out.get("consistency_loss", 0))
                 else:
                     if "mrnn" in self.hparams.imputator_name:
                         imputed = out[2]
                         target = batch["target"]
                         mask = batch["input_mask"]
                         imp_loss = torch.abs(imputed - target) * (1 - mask)
                         imp_loss = imp_loss.sum() / (1 - mask).sum().clamp(min=1)
                     else:
                         imputed = out[0]
                         imp_loss = out[-1] + out[-2]
            else:
                 out = self.imputer._forward_model(inputs, batch)
                 if "mrnn" in self.hparams.imputator_name:
                     imputed = out["imputed_data"]
                     target = batch["target"]
                     mask = batch["input_mask"]
                     imp_loss = torch.abs(imputed - target) * (1 - mask)
                     imp_loss = imp_loss.sum() / (1 - mask).sum().clamp(min=1)
                 else:
                     imputed = out["imputation"]
                     imp_loss = out["reconstruction_loss"] + out["consistency_loss"]
                     
            return imputed, imp_loss
        else:
            raise RuntimeError("Could not find suitable imputer interface.")

    def forward(self, batch):
        imputed, _ = self.get_imputer_outputs_and_loss(batch)
        orig_mask = batch["orig_mask"] if "orig_mask" in batch else batch["input_mask"]
        
        classifier_in = torch.cat([imputed, orig_mask], dim=-1)
        logits = self.classifier(classifier_in)
        return logits, imputed

    def training_step(self, batch, batch_idx):
        self._sync_device()
        imputed, imp_loss = self.get_imputer_outputs_and_loss(batch, is_training=True)
        orig_mask = batch["orig_mask"] if "orig_mask" in batch else batch["input_mask"]
        
        classifier_in = torch.cat([imputed, orig_mask], dim=-1)
        logits = self.classifier(classifier_in)
        
        labels = batch["label"].squeeze(-1)
        cls_loss = self.criterion_cls(logits, labels)
        loss = self.alpha * imp_loss + self.beta * cls_loss
        
        self.log("train/loss_total", loss, prog_bar=True)
        self.log("train/loss_imp", imp_loss, prog_bar=True)
        self.log("train/loss_cls", cls_loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        self._sync_device()
        imputed, imp_loss = self.get_imputer_outputs_and_loss(batch)
        orig_mask = batch["orig_mask"] if "orig_mask" in batch else batch["input_mask"]
        
        classifier_in = torch.cat([imputed, orig_mask], dim=-1)
        logits = self.classifier(classifier_in)
        
        labels = batch["label"].squeeze(-1)
        cls_loss = self.criterion_cls(logits, labels)
        loss = self.alpha * imp_loss + self.beta * cls_loss
        
        probs = torch.sigmoid(logits)
        self.val_preds.append(probs.detach().cpu())
        self.val_targets.append(labels.detach().cpu())
        
        self.log("val/loss", loss)
        self.log("val/loss_cls", cls_loss)
        
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()
        
        if len(np.unique(targets)) > 1:
            auroc = roc_auc_score(targets, preds)
            auprc = average_precision_score(targets, preds)
            self.log("val/auroc", auroc, prog_bar=True)
            self.log("val/auprc", auprc, prog_bar=True)
            
        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        # Use imputer's own lr for fine-tuning the foundation model
        imputer_lr = getattr(self.imputer, 'lr', self.lr)
        imputer_wd =  getattr(self.imputer, 'weight_decay', self.weight_decay)
        
        optimizer = torch.optim.Adam([
            {"params": self.imputer.parameters(), "lr": imputer_lr, "weight_decay": imputer_wd},
            {"params": self.classifier.parameters(), "lr": self.lr, "weight_decay": self.weight_decay}
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/auprc",
            },
        }
