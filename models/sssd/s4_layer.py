"""
Simplified S4 Layer — Vendored from state-spaces/s4 concepts

Implements the core Structured State Space (S4) layer for sequence modeling.
This is a simplified but functional version suitable for the SSSD denoising
backbone.

S4 models sequences via a continuous-time state space:
    x'(t) = A x(t) + B u(t)
    y(t)  = C x(t) + D u(t)

Discretized and computed efficiently via convolution in the frequency domain.

Reference:
    Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces"
    (ICLR 2022)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class S4Layer(nn.Module):
    """
    Simplified S4 Layer.

    Uses a diagonal approximation of the HiPPO-LegS matrix for efficiency.
    Computes the state space convolution in the frequency domain via FFT.

    Args:
        d_model: input/output channel dimension
        state_dim: state space dimension N (default 128)
        dropout: dropout rate (default 0.2)
        bidirectional: if True, use bidirectional processing (default False)
        layer_norm: if True, apply layer norm (default False)
        seq_len: maximum sequence length (for pre-computing kernel)
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 128,
        dropout: float = 0.2,
        bidirectional: bool = False,
        layer_norm: bool = False,
        seq_len: int = 48,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.bidirectional = bidirectional
        self.seq_len = seq_len

        # Learnable parameters for the diagonal state space model
        # A is parameterized as negative log-space for stability
        # Shift mean to +1.0 so exp(log_A_real) ≈ 2.7 -> A_diag ≈ -2.7 (strongly contractive)
        self.log_A_real = nn.Parameter(torch.randn(d_model, state_dim) * 0.5 + 1.0)
        self.A_imag = nn.Parameter(torch.randn(d_model, state_dim) * 0.1)

        # B and C are complex-valued
        self.B_real = nn.Parameter(torch.randn(d_model, state_dim) * (1.0 / state_dim))
        self.B_imag = nn.Parameter(torch.randn(d_model, state_dim) * (1.0 / state_dim))
        self.C_real = nn.Parameter(torch.randn(d_model, state_dim) * (1.0 / state_dim))
        self.C_imag = nn.Parameter(torch.randn(d_model, state_dim) * (1.0 / state_dim))

        # D (skip connection) — real valued
        self.D = nn.Parameter(torch.zeros(d_model))

        # Discretization step size (learnable)
        self.log_dt = nn.Parameter(torch.randn(d_model) * 0.01 - 1.0)

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model) if layer_norm else nn.Identity()

        # For bidirectional
        if bidirectional:
            self.out_proj = nn.Linear(d_model * 2, d_model)
        else:
            self.out_proj = nn.Linear(d_model, d_model)
        
        # Initialize projections to be small
        nn.init.kaiming_normal_(self.out_proj.weight, nonlinearity="linear")
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _compute_kernel(self, L: int) -> torch.Tensor:
        """
        Compute the SSM convolution kernel of length L.

        Uses the diagonal approximation:
            K[k] = C * A^k * B for k = 0, ..., L-1

        Computed efficiently via:
            K[k] = Re(C * diag(Ā^k) * B)

        where Ā = exp(A * dt) is the discretized state matrix.

        Returns:
            kernel: (d_model, L) real-valued convolution kernel
        """
        dt = torch.exp(self.log_dt)  # (d_model,)

        # Continuous-time A (diagonal, complex)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (d_model, N)

        # Discretize: Ā = exp(A * dt)
        A_bar = torch.exp(A * dt.unsqueeze(-1))  # (d_model, N)

        # B and C as complex
        B = self.B_real + 1j * self.B_imag  # (d_model, N)
        C = self.C_real + 1j * self.C_imag  # (d_model, N)

        # Compute C * Ā^k * B for k=0..L-1
        # Ā^k = Ā^0, Ā^1, ..., Ā^(L-1)
        # Use vandermonde-like computation
        powers = torch.arange(L, device=A.device, dtype=torch.float32)  # (L,)
        # (d_model, N, 1) ** (1, 1, L) → (d_model, N, L)
        A_powers = A_bar.unsqueeze(-1) ** powers.unsqueeze(0).unsqueeze(0)

        # kernel[k] = sum_n C[n] * A^k[n] * B[n]
        # (d_model, N) * (d_model, N, L) * (d_model, N) → sum over N → (d_model, L)
        CB = (C * B).unsqueeze(-1)  # (d_model, N, 1)
        kernel = (CB * A_powers).sum(dim=1)  # (d_model, L)

        # Take real part and scale by dt
        kernel = kernel.real * dt.unsqueeze(-1)  # (d_model, L)

        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, channels, length) — channel-first format

        Returns:
            y: (batch, channels, length)
        """
        B, C, L = x.shape
        residual = x

        # Compute convolution kernel
        kernel = self._compute_kernel(L)  # (d_model, L)

        # FFT-based convolution for efficiency
        # Pad to avoid circular convolution artifacts
        fft_len = 2 * L
        x_fft = torch.fft.rfft(x, n=fft_len, dim=-1)         # (B, C, fft_len//2+1)
        k_fft = torch.fft.rfft(kernel, n=fft_len, dim=-1)     # (C, fft_len//2+1)
        y_fft = x_fft * k_fft.unsqueeze(0)                    # (B, C, fft_len//2+1)
        y = torch.fft.irfft(y_fft, n=fft_len, dim=-1)[..., :L]  # (B, C, L)

        # Add skip connection (D term)
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * x  # (B, C, L)

        if self.bidirectional:
            # Process reversed sequence
            x_flip = torch.flip(x, dims=[-1])
            x_flip_fft = torch.fft.rfft(x_flip, n=fft_len, dim=-1)
            y_flip_fft = x_flip_fft * k_fft.unsqueeze(0)
            y_flip = torch.fft.irfft(y_flip_fft, n=fft_len, dim=-1)[..., :L]
            y_flip = torch.flip(y_flip, dims=[-1])
            y_flip = y_flip + self.D.unsqueeze(0).unsqueeze(-1) * x

            # Combine forward and backward
            y = torch.cat([y, y_flip], dim=1)  # (B, 2C, L)

        # Rearrange for normalization: (B, C, L) → (B, L, C) → norm → (B, C, L)
        y = rearrange(y, "b c l -> b l c")
        y = self.norm(y)
        y = self.out_proj(y)
        y = self.dropout(y)
        y = rearrange(y, "b l c -> b c l")

        # Residual connection
        y = y + residual

        return y


class S4Block(nn.Module):
    """
    S4 Block: Two S4 layers with GELU activation in between.

    Used as the building block for SSSD residual layers.

    Args:
        d_model: channel dimension
        state_dim: S4 state dimension
        dropout: dropout rate
        bidirectional: bidirectional S4
        layer_norm: apply layer norm in S4
        seq_len: maximum sequence length
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 128,
        dropout: float = 0.2,
        bidirectional: bool = False,
        layer_norm: bool = False,
        seq_len: int = 48,
    ):
        super().__init__()
        self.s4_1 = S4Layer(
            d_model=d_model, state_dim=state_dim, dropout=dropout,
            bidirectional=bidirectional, layer_norm=layer_norm, seq_len=seq_len,
        )
        self.s4_2 = S4Layer(
            d_model=d_model, state_dim=state_dim, dropout=dropout,
            bidirectional=bidirectional, layer_norm=layer_norm, seq_len=seq_len,
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length)
        Returns:
            (batch, channels, length)
        """
        x = self.s4_1(x)
        x = self.activation(rearrange(x, "b c l -> b l c"))
        x = self.norm(x)
        x = rearrange(x, "b l c -> b c l")
        x = self.s4_2(x)
        return x
