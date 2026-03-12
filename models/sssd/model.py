"""
SSSD LightningModule — Diffusion-based Imputation with S4 Backbone

Full model from Alcaraz & Strodthoff, "Diffusion-based Time Series
Imputation and Forecasting with Structured State Space Models" (TMLR 2023).

Architecture:
    1. Input: concat(noisy_data, observed_data, mask) → input projection
    2. Diffusion step embedding → added to each residual layer
    3. 36 residual layers, each with 2 S4 layers
    4. Skip connections aggregated → output projection → predicted noise

The model predicts the noise ε added during the forward diffusion,
and the loss is MSE between predicted and true noise.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import os

from metrics.imputation import mae, rmse
from models.sssd.diffusion import (
    DiffusionSchedule,
    DiffusionStepEmbedding,
    conditional_q_sample,
    conditional_q_sample,
    p_sample_loop,
    p_sample_loop_accelerated,
)
from models.sssd.s4_layer import S4Block


class SSSDResidualLayer(nn.Module):
    """
    Single SSSD residual layer. Supports optional DGI v2 semantic gating.
    """

    def __init__(
        self,
        residual_channels: int = 256,
        skip_channels: int = 256,
        diffusion_embed_dim: int = 256,
        s4_state_dim: int = 128,
        s4_dropout: float = 0.2,
        seq_len: int = 48,
        kgi_injector=None,
        kgi_gate=None
    ):
        super().__init__()
        self.kgi_injector = kgi_injector
        self.kgi_gate = kgi_gate

        # S4 block
        self.s4_block = S4Block(
            d_model=residual_channels,
            state_dim=s4_state_dim,
            dropout=s4_dropout,
            bidirectional=False,
            layer_norm=False,
            seq_len=seq_len,
        )

        # Diffusion step conditioning
        self.diffusion_proj = nn.Linear(diffusion_embed_dim, residual_channels)

        # Output projections
        self.res_proj = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_proj = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

        self.norm = nn.GroupNorm(8, residual_channels)
        
        # Zero-initialize the residual projection to make the block an identity initially
        nn.init.zeros_(self.res_proj.weight)
        if self.res_proj.bias is not None:
            nn.init.zeros_(self.res_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        diffusion_emb: torch.Tensor,
        mask: torch.Tensor = None,
        medbert_dict: dict = None,
        kgi_itemids: list = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(x)

        # Add diffusion step conditioning
        d = self.diffusion_proj(diffusion_emb)  # (B, C)
        h = h + d.unsqueeze(-1)  # broadcast over length

        # S4 processing
        h = self.s4_block(h)

        # --- DGI v2: Optional Semantic Gating ---
        if self.kgi_gate is not None and self.kgi_injector is not None and medbert_dict is not None:
            # h is [B, C, T], need [B, T, C] for gate
            h_time = h.transpose(1, 2)
            
            # Retrieve semantic context
            kgi_context = self.kgi_injector(
                query_hidden=h_time, 
                surviving_mask=mask, 
                precomputed_embeddings=medbert_dict, 
                variable_indices=kgi_itemids
            )
            
            # Apply Mask-Aware Gating (v2) if enabled, else v1
            mask_agg = mask.float().mean(dim=-1, keepdim=True) if mask is not None else None
            h_fused = self.kgi_gate(h_time, kgi_context, mask=mask_agg)
            
            h = h_fused.transpose(1, 2)

        # Split into residual and skip
        res = self.res_proj(h)
        skip = self.skip_proj(h)

        # Residual connection (Original logic preserved)
        res = (res + x) / (2 ** 0.5)

        return res, skip


class SSSDDenoiser(nn.Module):
    """
    SSSD Denoising Network. Supports optional Knowledge Graph Injection.
    """

    def __init__(
        self,
        d_feature: int = 9,
        residual_layers: int = 36,
        residual_channels: int = 256,
        skip_channels: int = 256,
        diffusion_embed_dim: int = 256,
        s4_state_dim: int = 128,
        s4_dropout: float = 0.2,
        seq_len: int = 48,
        use_graph_prior: bool = True,
        use_kgi: bool = False,
        kgi_mode: str = "dgi_mask",
    ):
        super().__init__()
        self.d_feature = d_feature
        self.use_kgi = use_kgi
        
        # KGI Components (Conditional)
        if self.use_kgi:
            from models.saits.layers import FeatureContextualGate
            from models.saits.kgi_layer import DynamicKnowledgeInjector
            self.kgi_injector = DynamicKnowledgeInjector(text_embed_dim=768, hidden_dim=residual_channels)
            self.kgi_gate = FeatureContextualGate(residual_channels, mask_aware=(kgi_mode == "dgi_mask"))
        else:
            self.kgi_injector = None
            self.kgi_gate = None

        self.use_graph_prior = use_graph_prior and (d_feature == 17)
        input_dim_multiplier = 4 if self.use_graph_prior else 3
        
        self.input_proj = nn.Conv1d(d_feature * input_dim_multiplier, residual_channels, kernel_size=1)
        self.diffusion_embedding = DiffusionStepEmbedding(diffusion_embed_dim)

        # Residual layers (passing KGI references if enabled)
        self.residual_layers = nn.ModuleList([
            SSSDResidualLayer(
                residual_channels=residual_channels,
                skip_channels=skip_channels,
                diffusion_embed_dim=diffusion_embed_dim,
                s4_state_dim=s4_state_dim,
                s4_dropout=s4_dropout,
                seq_len=seq_len,
                kgi_injector=self.kgi_injector,
                kgi_gate=self.kgi_gate
            )
            for _ in range(residual_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, skip_channels),
            nn.GELU(),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(skip_channels, d_feature, kernel_size=1),
        )

        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        M: torch.Tensor = None,
        medbert_dict: dict = None,
        kgi_itemids: list = None
    ) -> torch.Tensor:
        B, T, D = x_noisy.shape

        if self.use_graph_prior and M is not None:
            x_graph = torch.matmul(observed, M.T)
            x_input = torch.cat([x_noisy, observed, mask, x_graph], dim=-1)
        else:
            x_input = torch.cat([x_noisy, observed, mask], dim=-1)

        x_input = x_input.permute(0, 2, 1)
        h = self.input_proj(x_input)
        diff_emb = self.diffusion_embedding(t)

        skip_sum = torch.zeros_like(h[:, :self.output_proj[2].in_channels, :])

        for layer in self.residual_layers:
            # Conditional pass based on use_kgi
            if self.use_kgi:
                h, skip = layer(h, diff_emb, mask=mask, medbert_dict=medbert_dict, kgi_itemids=kgi_itemids)
            else:
                h, skip = layer(h, diff_emb)
            skip_sum = skip_sum + skip

        skip_sum = skip_sum / (len(self.residual_layers) ** 0.5)
        out = self.output_proj(skip_sum)
        predicted_noise = out.permute(0, 2, 1)

        return predicted_noise


class SSSDModule(pl.LightningModule):
    """
    SSSD: Structured State Space Diffusion Model for Imputation.

    Training:
        1. Sample random diffusion timestep t
        2. Apply conditional forward diffusion (noise only on imputation targets)
        3. Predict noise with denoiser
        4. MSE loss between predicted and true noise

    Inference:
        Full reverse diffusion loop (1000 steps, or accelerated with DDIM).

    Args:
        d_feature: number of features D
        residual_layers: number of S4 residual layers
        residual_channels: channel dim
        skip_channels: skip connection dim
        diffusion_embed_dim: timestep embedding dim
        s4_state_dim: S4 state dimension
        s4_dropout: S4 dropout
        T: number of diffusion steps
        beta_start: noise schedule start
        beta_end: noise schedule end
        lr: learning rate
        seq_len: sequence length
    """

    def __init__(
        self,
        d_feature: int = 9,
        residual_layers: int = 36,
        residual_channels: int = 256,
        skip_channels: int = 256,
        diffusion_embed_dim: int = 256,
        s4_state_dim: int = 128,
        s4_dropout: float = 0.2,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        seq_len: int = 48,
        inference_samples: int = 5,
        inference_steps: int = 1000,
        use_graph_prior: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.inference_samples = inference_samples
        self.inference_steps = inference_steps

        # Diffusion schedule
        self.schedule = DiffusionSchedule(T=T, beta_start=beta_start, beta_end=beta_end)

        # Denoising network
        self.denoiser = SSSDDenoiser(
            d_feature=d_feature,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            diffusion_embed_dim=diffusion_embed_dim,
            s4_state_dim=s4_state_dim,
            s4_dropout=s4_dropout,
            seq_len=seq_len,
            use_graph_prior=use_graph_prior,
        )
        
        # Load Meta-Path Prior
        self.use_graph_prior = use_graph_prior and (d_feature == 17)
        if self.use_graph_prior:
            prior_path = "graph/artifacts_pruned/metapath_prior.npy"
            if os.path.exists(prior_path):
                print("SSSD: Using Meta-Path Conditioning (D=17)")
                M = np.load(prior_path)
                self.register_buffer("metapath_prior", torch.from_numpy(M).float())
            else:
                print(f"Warning: Meta-path prior not found at {prior_path}. Using identity.")
                self.register_buffer("metapath_prior", torch.eye(d_feature))
        else:
            self.metapath_prior = None

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Inference: run full reverse diffusion to impute missing values.

        Args:
            batch: dict with data, orig_mask, artificial_mask, input_mask

        Returns:
            imputed: (B, T, D) imputed values
        """
        data = batch["data"]
        input_mask = batch["input_mask"]
        observed = data * input_mask

        def model_fn(x_t, t, obs, m):
            return self.denoiser(x_t, t, obs, m, self.metapath_prior)

        imputed = p_sample_loop(
            model_fn=model_fn,
            shape=data.shape,
            schedule=self.schedule,
            observed=observed,
            mask=input_mask,
            device=self.device,
            n_samples=self.inference_samples,
        )

        return imputed

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        data = batch["data"]                    # (B, T, D) ground truth
        input_mask = batch["input_mask"]        # (B, T, D) 1=visible to model
        artificial_mask = batch["artificial_mask"]  # (B, T, D) 1=imputation target

        observed = data * input_mask
        B = data.shape[0]

        # Sample random diffusion timesteps
        t = torch.randint(0, self.schedule.T, (B,), device=self.device)

        # Conditional forward diffusion: noise only on imputation targets
        x_t, noise = conditional_q_sample(
            x_0=data,
            observed=observed,
            mask=input_mask,
            schedule=self.schedule,
            t=t,
        )

        # Predict noise
        predicted_noise = self.denoiser(x_t, t, observed, input_mask, self.metapath_prior)

        # Full Sequence Loss: Compute MSE over all positions to ensure temporal stability
        loss = ((predicted_noise - noise) ** 2).mean()

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        data = batch["data"]
        input_mask = batch["input_mask"]
        artificial_mask = batch["artificial_mask"]
        observed = data * input_mask

        B = data.shape[0]

        # Training-style loss (fast)
        t = torch.randint(0, self.schedule.T, (B,), device=self.device)
        x_t, noise = conditional_q_sample(
            x_0=data, observed=observed, mask=input_mask,
            schedule=self.schedule, t=t,
        )
        predicted_noise = self.denoiser(x_t, t, observed, input_mask, self.metapath_prior)
        
        # Full Sequence Loss
        loss = ((predicted_noise - noise) ** 2).mean()

        self.log("val/loss", loss, prog_bar=True)

        # Full imputation only on first batch (expensive)
        if batch_idx == 0:
            with torch.no_grad():
                # Use fewer samples for speed during validation
                def model_fn(x_t, t, obs, m):
                    return self.denoiser(x_t, t, obs, m, self.metapath_prior)

                if self.inference_steps < self.schedule.T:
                    imputed = p_sample_loop_accelerated(
                        model_fn=model_fn,
                        shape=data.shape,
                        schedule=self.schedule,
                        observed=observed,
                        mask=input_mask,
                        device=self.device,
                        n_samples=1,
                        inference_steps=self.inference_steps,
                    )
                else:
                    imputed = p_sample_loop(
                        model_fn=model_fn,
                        shape=data.shape,
                        schedule=self.schedule,
                        observed=observed,
                        mask=input_mask,
                        device=self.device,
                        n_samples=1,
                    )

                val_mae = mae(imputed, data, artificial_mask)
                val_rmse = rmse(imputed, data, artificial_mask)

                self.log("val/mae", val_mae, prog_bar=True)
                self.log("val/rmse", val_rmse)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        data = batch["data"]
        input_mask = batch["input_mask"]
        artificial_mask = batch["artificial_mask"]
        observed = data * input_mask

        def model_fn(x_t, t, obs, m):
            return self.denoiser(x_t, t, obs, m, self.metapath_prior)

        imputed = p_sample_loop(
            model_fn=model_fn,
            shape=data.shape,
            schedule=self.schedule,
            observed=observed,
            mask=input_mask,
            device=self.device,
            n_samples=self.inference_samples,
        )

        test_mae = mae(imputed, data, artificial_mask)
        test_rmse = rmse(imputed, data, artificial_mask)

        self.log("test/mae", test_mae, sync_dist=True)
        self.log("test/rmse", test_rmse, sync_dist=True)

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
