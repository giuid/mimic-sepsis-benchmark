import torch
import torch.nn as nn
import pytorch_lightning as pl
from pypots.imputation import GPVAE as PyPOTSGPVAE
from metrics.imputation import (
    mae, rmse, mre, r2_score, correlation_error
)

class GPVAEModule(pl.LightningModule):
    def __init__(
        self,
        d_feature: int,
        seq_len: int,
        latent_size: int = 64,
        encoder_sizes: list = [128, 128],
        decoder_sizes: list = [128, 128],
        lr: float = 1e-3,
        weight_decay: float = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.pypots_model = PyPOTSGPVAE(
            n_steps=seq_len,
            n_features=d_feature,
            latent_size=latent_size,
            encoder_sizes=encoder_sizes,
            decoder_sizes=decoder_sizes,
            epochs=1,
        )
        self.model = self.pypots_model.model
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, batch):
        x = batch["data"]
        mask = batch["input_mask"]
        inputs = {
            "X": x,
            "missing_mask": (1 - mask).bool()
        }
        output_data = self.model(inputs, calc_criterion=False)
        imputed = output_data.get("imputation", output_data.get("reconstruction"))
        if imputed is not None and imputed.ndim == 4:
            imputed = imputed.squeeze(1)
        return imputed

    def training_step(self, batch, batch_idx):
        x = batch["data"]
        mask = batch["input_mask"]
        missing_mask = (1 - mask).bool()
        
        inputs = {
            "X": x,
            "missing_mask": missing_mask
        }
        
        # PyPOTS core might need explicit train mode and calc_criterion=True to return 'loss'
        self.model.train()
        output = self.model(inputs, calc_criterion=True)
        
        # Robust loss extraction: PyPOTS uses 'loss' in train, 'metric' in eval
        loss = output.get("loss", output.get("metric"))
        
        if loss is None:
            # Fallback for GP-VAE if it returns sub-keys
            loss = output.get("nll", output.get("elbo"))
            
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=self.device, requires_grad=True)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["data"]
        mask = batch["input_mask"]
        missing_mask = (1 - mask).bool()
        
        inputs = {
            "X": x,
            "missing_mask": missing_mask
        }
        # In validation, self.model should be in eval() mode
        output_loss = self.model(inputs, calc_criterion=True)
        # Separate pass to get imputed data for MAE calculation
        output_data = self.model(inputs, calc_criterion=False)
        
        loss = output_loss.get("metric", output_loss.get("loss"))
        
        if loss is None:
            loss = torch.tensor(0.0, device=self.device)
            
        # Compute eval metrics for comparison with SAITS
        imputed = output_data.get("imputation", output_data.get("reconstruction"))
        if imputed is not None:
            target = batch["target"]
            eval_mask = batch["artificial_mask"]
            
            try:
                val_mae = mae(imputed, target, eval_mask)
                val_rmse = rmse(imputed, target, eval_mask)
                val_mre = mre(imputed, target, eval_mask)
                
                self.log("val/mae", val_mae, prog_bar=True)
                self.log("val/rmse", val_rmse, prog_bar=True)
                self.log("val/mre", val_mre)
            except Exception as e:
                print(f"GPVAE MAE error: {e}")

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> None:
        # Re-run same logic as validation but for test metrics
        inputs = {
            "X": batch["data"],
            "missing_mask": (1 - batch["input_mask"]).bool()
        }
        output_data = self.model(inputs, calc_criterion=False)
        imputed = output_data.get("imputation", output_data.get("reconstruction"))
        
        if imputed is not None:
            target = batch["target"]
            eval_mask = batch["artificial_mask"]
            
            try:
                test_mae = mae(imputed, target, eval_mask)
                test_rmse = rmse(imputed, target, eval_mask)
                test_mre = mre(imputed, target, eval_mask)
                test_r2 = r2_score(imputed, target, eval_mask)
                test_corr_err = correlation_error(imputed, target)
                
                self.log("test/mae", test_mae)
                self.log("test/rmse", test_rmse)
                self.log("test/mre", test_mre)
                self.log("test/r2", test_r2)
                self.log("test/corr_err", test_corr_err)
            except Exception as e:
                print(f"GPVAE Test error: {e}")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
