"""
Diffusion Utilities for SSSD

Implements the forward and reverse diffusion processes for conditional
time series imputation, following the DDPM framework.

Key concepts:
    Forward process:  q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I)
    Reverse process:  p_θ(x_{t-1} | x_t) learned by the denoising network
    Loss:             E[||ε - ε_θ(x_t, t)||²]

For imputation (D1 setup):
    - Noise is applied ONLY to the positions that need imputation
    - Observed positions are kept fixed throughout the diffusion
    - The model is conditioned on (noisy_target, observed_values, mask)

Reference:
    Alcaraz & Strodthoff, "Diffusion-based Time Series Imputation and
    Forecasting with Structured State Space Models" (TMLR 2023)
"""

import torch
import torch.nn as nn
import numpy as np


class DiffusionSchedule(nn.Module):
    """
    Linear noise schedule for DDPM.

    Precomputes all α, ᾱ, β, σ values needed for forward/reverse processes.

    Args:
        T: number of diffusion steps
        beta_start: starting noise level
        beta_end: ending noise level
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.T = T

        # Linear schedule
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), alpha_bars[:-1]])

        # Posterior variance: σ²_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        # Register as buffers (not parameters, but moved to device with model)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bars", alpha_bars.float())
        self.register_buffer("alpha_bars_prev", alpha_bars_prev.float())
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars).float())
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars).float())
        self.register_buffer("sqrt_recip_alphas", (1.0 / torch.sqrt(alphas)).float())
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer(
            "posterior_log_variance",
            torch.log(torch.clamp(posterior_variance, min=1e-20)).float(),
        )

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: sample x_t from q(x_t | x_0).

            x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε

        Args:
            x_0: (B, ...) clean data
            t: (B,) integer timesteps in [0, T-1]
            noise: (B, ...) optional pre-sampled noise

        Returns:
            x_t: (B, ...) noisy data
            noise: (B, ...) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ab = self.sqrt_alpha_bars[t]        # (B,)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t]  # (B,)

        # Reshape for broadcasting: (B,) → (B, 1, 1, ...)
        while sqrt_ab.dim() < x_0.dim():
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_omab = sqrt_omab.unsqueeze(-1)

        x_t = sqrt_ab * x_0 + sqrt_omab * noise
        return x_t, noise

    def p_mean_variance(
        self,
        predicted_noise: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of p_θ(x_{t-1} | x_t).

        Using the ε-prediction formulation:
            μ_θ = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ)

        Args:
            predicted_noise: ε_θ(x_t, t) from the denoising network
            x_t: (B, ...) current noisy data
            t: (B,) integer timesteps

        Returns:
            mean: (B, ...) predicted mean
            log_variance: (B, ...) log variance
        """
        beta = self.betas[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        sqrt_omab = self.sqrt_one_minus_alpha_bars[t]
        log_var = self.posterior_log_variance[t]

        # Reshape for broadcasting
        while beta.dim() < x_t.dim():
            beta = beta.unsqueeze(-1)
            sqrt_recip_alpha = sqrt_recip_alpha.unsqueeze(-1)
            sqrt_omab = sqrt_omab.unsqueeze(-1)
            log_var = log_var.unsqueeze(-1)

        mean = sqrt_recip_alpha * (x_t - (beta / sqrt_omab) * predicted_noise)

        return mean, log_var


def conditional_q_sample(
    x_0: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    schedule: DiffusionSchedule,
    t: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Conditional forward diffusion for imputation (D1 setup).

    Only applies noise to positions where mask=0 (to be imputed).
    Observed positions (mask=1) are kept as-is.

    Args:
        x_0: (B, T, D) original clean data (with 0 at missing positions)
        observed: (B, T, D) observed values
        mask: (B, T, D) 1=observed/visible, 0=to be imputed
        schedule: DiffusionSchedule instance
        t: (B,) diffusion timesteps
        noise: optional pre-sampled noise

    Returns:
        x_t: (B, T, D) conditionally noised data
        noise: (B, T, D) the added noise
    """
    x_t, noise = schedule.q_sample(x_0, t, noise)

    # Keep observed positions fixed
    x_t = mask * observed + (1 - mask) * x_t

    return x_t, noise


@torch.no_grad()
def p_sample_loop(
    model_fn,
    shape: tuple,
    schedule: DiffusionSchedule,
    observed: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    n_samples: int = 1,
) -> torch.Tensor:
    """
    Full reverse diffusion loop for imputation.

    Starting from pure noise on imputation positions and observed values
    on known positions, iteratively denoise to produce imputations.

    Args:
        model_fn: callable(x_t, t, observed, mask) → predicted_noise
        shape: (B, T, D) output shape
        schedule: DiffusionSchedule
        observed: (B, T, D) observed values
        mask: (B, T, D) 1=observed, 0=impute
        device: torch device
        n_samples: number of samples to average (default 1)

    Returns:
        imputed: (B, T, D) imputed values (averaged over n_samples)
    """
    all_samples = []

    for _ in range(n_samples):
        # Start from noise on imputation positions
        x = torch.randn(shape, device=device)
        x = mask * observed + (1 - mask) * x

        for t_idx in reversed(range(schedule.T)):
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)

            # Model predicts noise
            pred_noise = model_fn(x, t, observed, mask)

            # Compute p(x_{t-1} | x_t)
            mean, log_var = schedule.p_mean_variance(pred_noise, x, t)

            if t_idx > 0:
                noise = torch.randn_like(x)
                x = mean + torch.exp(0.5 * log_var) * noise
            else:
                x = mean

            # Re-apply conditioning: keep observed values fixed
            x = mask * observed + (1 - mask) * x

        all_samples.append(x)

    # Average over samples
    imputed = torch.stack(all_samples, dim=0).mean(dim=0)
    return imputed


class DiffusionStepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding for diffusion models.

    Maps integer timestep t to a continuous embedding vector.

    Args:
        embed_dim: output embedding dimension
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer timesteps

        Returns:
            embedding: (B, embed_dim)
        """
        half_dim = self.embed_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.fc(emb)


@torch.no_grad()
def p_sample_loop_accelerated(
    model_fn,
    shape: tuple,
    schedule: DiffusionSchedule,
    observed: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    n_samples: int = 1,
    inference_steps: int = 50,
    eta: float = 0.0,  # 0.0 = DDIM (deterministic), 1.0 = DDPM
) -> torch.Tensor:
    """
    Accelerated reverse diffusion (DDIM/Strided Sampling).
    
    Args:
        eta: Weight of noise (0 for pure DDIM, 1 for DDPM-like)
    """
    all_samples = []
    
    # Create strided timesteps (e.g., 1000, 980, 960, ...)
    # We want inference_steps steps distributed along [0, T-1]
    # np.linspace guarantees we hit the endpoints roughly
    times = torch.linspace(schedule.T - 1, 0, inference_steps + 1).long().to(device)
    # times = [999, 979, ..., 19, 0] roughly (if linear)
    # Actually we just use a subset directly
    
    # Let's use a simple stride
    step_size = schedule.T // inference_steps
    timesteps = list(range(0, schedule.T, step_size))
    timesteps = sorted(timesteps, reverse=True) # [980, 960, ..., 0] if T=1000, steps=50 (step=20)
    # Ensure 0 is included if we want to go to the end, but usually we just start from T-1
    # Better approach:
    # timesteps = torch.linspace(schedule.T - 1, 0, inference_steps).long().tolist()
    
    # Robust striding
    timesteps = np.linspace(schedule.T - 1, 0, inference_steps).astype(int)
    
    for _ in range(n_samples):
        # Start from noise
        x = torch.randn(shape, device=device)
        x = mask * observed + (1 - mask) * x
        
        for i, t_idx in enumerate(timesteps):
            # Current timestep t
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            
            # Next timestep t_prev (the one we are stepping TO)
            # If this is the last step (i == len-1), t_prev is conceptually -1 (fully denoised)
            # But usually we strictly stop at 0.
            if i == len(timesteps) - 1:
                t_prev_idx = -1
            else:
                t_prev_idx = timesteps[i + 1]
                
            # 1. Predict noise ε_θ(x_t)
            pred_noise = model_fn(x, t, observed, mask)
            
            # 2. Estimate x_0 (predicted clean data)
            # x_0 = (x_t - √1-ᾱ_t * ε) / √ᾱ_t
            alpha_bar_t = schedule.alpha_bars[t_idx]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
            
            pred_x0 = (x - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
            
            # 3. Compute direction to x_{t_prev}
            if t_prev_idx >= 0:
                alpha_bar_prev = schedule.alpha_bars[t_prev_idx]
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
                )
                
                # DDIM direction
                dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * pred_noise
                
                # Noise term
                noise = torch.randn_like(x) if sigma_t > 0 else 0.0
                
                x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise
            else:
                # Last step: just output x_0
                x_prev = pred_x0
            
            x = x_prev
            
            # Re-apply conditioning
            x = mask * observed + (1 - mask) * x
            
        all_samples.append(x)
        
    imputed = torch.stack(all_samples, dim=0).mean(dim=0)
    return imputed
