"""
Imputation Metrics

MAE and RMSE computed only on artificially masked positions.
Both global and per-variable variants.
"""

import numpy as np
import torch


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)) and len(x) > 0:
        # If it's a list/tuple, try to find the actual data tensor
        # This is a fallback for PyPOTS models that return nested structures
        for item in x:
            if isinstance(item, (torch.Tensor, np.ndarray)):
                return to_np(item)
    return x

def mae(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
) -> float:
    """
    Mean Absolute Error on masked positions.
    """
    try:
        pred, target, mask = to_np(pred), to_np(target), to_np(mask)
        if not isinstance(pred, np.ndarray):
            return 9.999
            
        mask = mask.astype(bool)
        if mask.sum() == 0:
            return 0.0

        return float(np.abs(pred[mask] - target[mask]).mean())
    except Exception as e:
        s_pred = pred.shape if hasattr(pred, "shape") else "vals=" + str(pred)[:50]
        s_targ = target.shape if hasattr(target, "shape") else "vals=" + str(target)[:50]
        s_mask = mask.shape if hasattr(mask, "shape") else "vals=" + str(mask)[:50]
        print(f"MAE Error: {e} | Shapes: pred={s_pred}, target={s_targ}, mask={s_mask}")
        return 9.999

def rmse(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
) -> float:
    """
    Root Mean Squared Error on masked positions.
    """
    try:
        pred, target, mask = to_np(pred), to_np(target), to_np(mask)
        if not isinstance(pred, np.ndarray):
            return 9.999
            
        mask = mask.astype(bool)
        if mask.sum() == 0:
            return 0.0

        return float(np.sqrt(((pred[mask] - target[mask]) ** 2).mean()))
    except Exception as e:
        return 9.999

def mre(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Mean Relative Error (MRE) on masked positions.
    """
    try:
        pred, target, mask = to_np(pred), to_np(target), to_np(mask)
        if not isinstance(pred, np.ndarray):
            return 9.999
            
        mask = mask.astype(bool)
        if mask.sum() == 0:
            return 0.0

        relative_diff = np.abs(pred[mask] - target[mask]) / (np.abs(target[mask]) + eps)
        return float(relative_diff.mean())
    except Exception:
        return 9.999

def r2_score(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
) -> float:
    """
    R-squared (Coefficient of Determination) on masked positions.
    """
    try:
        pred, target, mask = to_np(pred), to_np(target), to_np(mask)
        if not isinstance(pred, np.ndarray):
            return -9.999
            
        mask = mask.astype(bool)
        if mask.sum() == 0:
            return 0.0

        p = pred[mask]
        t = target[mask]
        
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
            
        return float(1 - (ss_res / ss_tot))
    except Exception:
        return -9.999

def correlation_error(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
) -> float:
    """
    Correlation Error: mean absolute difference between feature correlation matrices.
    """
    try:
        pred, target = to_np(pred), to_np(target)
        if not isinstance(pred, np.ndarray):
            return 9.999

        # Flatten (N, T, D) -> (N*T, D)
        pred_flat = pred.reshape(-1, pred.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])
        
        import pandas as pd
        corr_pred = pd.DataFrame(pred_flat).corr().fillna(0).values
        corr_target = pd.DataFrame(target_flat).corr().fillna(0).values
        
        return float(np.abs(corr_pred - corr_target).mean())
    except Exception:
        return 9.999

def per_variable_metrics(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Per-variable metrics (MAE, RMSE, MRE, R2).
    """
    pred, target, mask = to_np(pred), to_np(target), to_np(mask)
    if not isinstance(pred, np.ndarray):
        return {}
        
    D = pred.shape[-1]
    if feature_names is None:
        feature_names = [f"feature_{d}" for d in range(D)]

    results = {}
    for d in range(D):
        m = mask[..., d].astype(bool)
        n_masked = int(m.sum())

        if n_masked == 0:
            results[feature_names[d]] = {
                "mae": 0.0, "rmse": 0.0, "mre": 0.0, "r2": 0.0, "n_masked": 0
            }
            continue

        p = pred[..., d][m]
        t = target[..., d][m]

        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

        results[feature_names[d]] = {
            "mae": float(np.abs(p - t).mean()),
            "rmse": float(np.sqrt(((p - t) ** 2).mean())),
            "mre": float((np.abs(p - t) / (np.abs(t) + 1e-8)).mean()),
            "r2": float(r2),
            "n_masked": n_masked,
        }

    return results


def mae_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable MAE loss on masked positions (for training).
    """
    masked_diff = torch.abs(pred - target) * mask
    n_masked = mask.sum()
    if n_masked == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return masked_diff.sum() / n_masked


def mse_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable MSE loss on masked positions (for training).
    """
    masked_diff = ((pred - target) ** 2) * mask
    n_masked = mask.sum()
    if n_masked == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return masked_diff.sum() / n_masked
