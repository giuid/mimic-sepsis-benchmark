import torch
import torch.nn as nn
import numpy as np

def test_s4_kernel():
    d_model = 256
    state_dim = 128
    L = 48
    
    # 1. Setup Parameters (mimic initialization)
    log_A_real = nn.Parameter(torch.randn(d_model, state_dim) * 0.5)
    A_imag = nn.Parameter(torch.randn(d_model, state_dim) * 0.1)
    B_real = nn.Parameter(torch.randn(d_model, state_dim) * (1.0 / state_dim))
    B_imag = nn.Parameter(torch.randn(d_model, state_dim) * (1.0 / state_dim))
    C_real = nn.Parameter(torch.randn(d_model, state_dim) * (1.0 / state_dim))
    C_imag = nn.Parameter(torch.randn(d_model, state_dim) * (1.0 / state_dim))
    log_dt = nn.Parameter(torch.randn(d_model) * 0.01 - 1.0)
    
    # 2. Compute Kernel
    dt = torch.exp(log_dt)
    A = -torch.exp(log_A_real) + 1j * A_imag
    A_bar = torch.exp(A * dt.unsqueeze(-1))
    
    # Check A_bar magnitude
    mag = torch.abs(A_bar)
    print(f"A_bar magnitude: min={mag.min().item():.6f}, max={mag.max().item():.6f}, mean={mag.mean().item():.6f}")
    
    B = B_real + 1j * B_imag
    C = C_real + 1j * C_imag
    
    powers = torch.arange(L, dtype=torch.float32)
    # A_powers = A_bar.unsqueeze(-1) ** powers.unsqueeze(0).unsqueeze(0)
    
    # Let's compare two ways of computing A_powers
    # Way 1: powers
    A_powers_1 = A_bar.unsqueeze(-1) ** powers.reshape(1, 1, -1)
    
    # Way 2: exp(log)
    A_powers_2 = torch.exp(torch.log(A_bar).unsqueeze(-1) * powers.reshape(1, 1, -1))
    
    diff = torch.abs(A_powers_1 - A_powers_2).mean()
    print(f"A_powers calculation diff: {diff.item():.2e}")
    
    # Check max A_powers magnitude
    mag_p = torch.abs(A_powers_1)
    print(f"A_powers magnitude: min={mag_p.min().item():.6f}, max={mag_p.max().item():.6f}, mean={mag_p.mean().item():.6f}")

    # Compute kernel
    CB = (C * B).unsqueeze(-1)
    kernel = (CB * A_powers_1).sum(dim=1)
    kernel = kernel.real * dt.unsqueeze(-1)
    
    print(f"Kernel range: min={kernel.min().item():.6f}, max={kernel.max().item():.6f}, mean={kernel.mean().item():.6f}, std={kernel.std().item():.6f}")
    
    # Estimate output variance for input with var=1
    # Var(y) = Sum(K^2) * Var(x)
    k_energy = (kernel**2).sum(dim=1).mean()
    print(f"Mean kernel energy (gain): {k_energy.item():.6f}")

if __name__ == "__main__":
    test_s4_kernel()
