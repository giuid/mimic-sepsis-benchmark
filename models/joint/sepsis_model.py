import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from data.dataset import configure_task

class SepsisClassifier(nn.Module):
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


class JointSepsisModule(pl.LightningModule):
    def __init__(
        self,
        imputator_name: str,
        imputator_kwargs: dict,
        d_feature: int = 22,
        task: str = 'ihm',
        alpha: float = 0.1,
        beta: float = 1.0,
        pos_weight: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.task = task.lower()
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.weight_decay = weight_decay
        
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
        else:
            raise ValueError(f"Unknown imputer {imputator_name}")
            
        # Downstream Classifier
        self.classifier = SepsisClassifier(input_dim=d_feature * 2) # X and Mask concatenated
        
        # Loss and Metrics from generic factory
        self.criterion_cls, self.val_metrics = configure_task(self.task, pos_weight)
        
        # Move metrics to device
        self.val_metrics = nn.ModuleDict(self.val_metrics)
        self.test_metrics = nn.ModuleDict({f"test_{k}": v.clone() for k, v in self.val_metrics.items()})

    def get_imputer_outputs_and_loss(self, batch, is_training: bool = False):
        # We assume dataset outputs {'data', 'input_mask', 'delta', 'label'}
        
        # BRITS requires specific missing targets
        if "brits" in self.hparams.imputator_name:
            if is_training and "artificial_mask" in batch and batch["artificial_mask"].sum() == 0:
                batch["indicating_mask"] = batch["input_mask"] 
            else:
                batch["indicating_mask"] = batch["input_mask"] # No missing target for downstream task
            batch["delta"] = batch.get("delta", torch.zeros_like(batch["data"]))

        # SAITS/MRNN base compatibility
        if "artificial_mask" not in batch:
             batch["indicating_mask"] = torch.zeros_like(batch["input_mask"])

        outputs = self.imputer(batch)
        
        imp_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self.imputer, '_compute_loss') and is_training:
            # We don't have explicit reconstruction targets natively from MIMICSepsisTaskDataset
            # unless we add continuous artificial missingness. Since the goal is DOWNSTREAM testing,
            # we can skip the imputation loss entirely or run it fully supervised on classification labels!
            # If the user wants joint training, we will need to inject artificial masks.
            pass
        
        # Identify imputed data
        if "imputed_data" in outputs:
            imputed = outputs["imputed_data"]
        elif "imputed_3" in outputs:
            imputed = outputs["imputed_3"]
        elif "reconstruction" in outputs:
            imputed = outputs["reconstruction"]
            
        return imputed, imp_loss

    def forward(self, batch):
        imputed, _ = self.get_imputer_outputs_and_loss(batch, is_training=False)
        # Connect to classifier
        # Sepsis baseline logic: concatenate input_mask
        class_input = torch.cat([imputed, batch["input_mask"]], dim=-1)
        logits = self.classifier(class_input)
        return logits

    def training_step(self, batch, batch_idx):
        imputed, imp_loss = self.get_imputer_outputs_and_loss(batch, is_training=True)
        class_input = torch.cat([imputed, batch["input_mask"]], dim=-1)
        
        outputs = self.classifier(class_input)
        
        # Ensure target matches output size (e.g., [128] vs [128, 1])
        target = batch["label"].squeeze()
        loss_cls = self.criterion_cls(outputs, target)
        
        total_loss = self.alpha * imp_loss + self.beta * loss_cls
        self.log("train/loss_cls", loss_cls, prog_bar=True)
        self.log("train/loss_imp", imp_loss)
        self.log("train/loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        target = batch["label"].squeeze()
        loss = self.criterion_cls(outputs, target)
        
        # Sepsis classifications use logits, continuous uses direct value
        if self.task in ['ihm', 'vr', 'ss']:
            preds = torch.sigmoid(outputs)
            targets = target.int()
        else:
            preds = outputs
            targets = target
            
        for name, metric in self.val_metrics.items():
            metric(preds, targets)
            self.log(f"val/{name}", metric, on_epoch=True, prog_bar=True)
            
        self.log("val/loss", loss)

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        target = batch["label"].squeeze()
        
        if self.task in ['ihm', 'vr', 'ss']:
            preds = torch.sigmoid(outputs)
            targets = target.int()
        else:
            preds = outputs
            targets = target
            
        for name, metric in self.test_metrics.items():
            metric(preds, targets)
            self.log(f"test/{name.replace('test_', '')}", metric, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
