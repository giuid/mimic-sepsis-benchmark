"""
Simple Imputation Baselines

Non-learned methods for comparison:
1. MeanImputer       — replace missing values with per-feature train mean
2. LOCFImputer       — Last Observation Carried Forward
3. LinearInterpImputer — Linear interpolation along time axis
"""

import numpy as np


class MeanImputer:
    """
    Mean Imputation: replace missing values with per-feature mean.

    The mean is computed from the training set (observed values only).
    At test time, imputation uses these pre-computed means.
    """

    def __init__(self):
        self.means = None

    def fit(self, data: np.ndarray, mask: np.ndarray) -> "MeanImputer":
        """
        Compute per-feature mean from observed values.

        Args:
            data: (N, T, D) training data (0 where missing)
            mask: (N, T, D) 1=observed

        Returns:
            self
        """
        D = data.shape[-1]
        self.means = np.zeros(D, dtype=np.float32)
        for d in range(D):
            observed = data[..., d][mask[..., d] == 1]
            if len(observed) > 0:
                self.means[d] = observed.mean()
        return self

    def impute(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Fill missing positions with per-feature mean.

        Args:
            data: (N, T, D) or (T, D) with 0s at missing positions
            mask: (N, T, D) or (T, D) where 1=observed

        Returns:
            imputed: same shape, missing positions filled with mean
        """
        assert self.means is not None, "Call fit() first"
        imputed = data.copy()
        for d in range(data.shape[-1]):
            missing = mask[..., d] == 0
            imputed[..., d][missing] = self.means[d]
        return imputed

    def __repr__(self) -> str:
        return "MeanImputer()"


class LOCFImputer:
    """
    Last Observation Carried Forward (LOCF).

    For each feature at each timestep, if the value is missing, use the
    most recent observed value. If no previous observation exists, use
    the per-feature mean (fallback).
    """

    def __init__(self):
        self.means = None

    def fit(self, data: np.ndarray, mask: np.ndarray) -> "LOCFImputer":
        """Compute per-feature means for fallback."""
        D = data.shape[-1]
        self.means = np.zeros(D, dtype=np.float32)
        for d in range(D):
            observed = data[..., d][mask[..., d] == 1]
            if len(observed) > 0:
                self.means[d] = observed.mean()
        return self

    def impute(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        LOCF imputation.

        Args:
            data: (N, T, D) with 0s at missing
            mask: (N, T, D) where 1=observed

        Returns:
            imputed: same shape
        """
        assert self.means is not None, "Call fit() first"

        if data.ndim == 2:
            return self._impute_single(data, mask)

        imputed = np.zeros_like(data)
        for i in range(data.shape[0]):
            imputed[i] = self._impute_single(data[i], mask[i])
        return imputed

    def _impute_single(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """LOCF for a single sample (T, D)."""
        T, D = data.shape
        imputed = data.copy()

        for d in range(D):
            last_val = self.means[d]  # fallback
            for t in range(T):
                if mask[t, d] == 1:
                    last_val = data[t, d]
                else:
                    imputed[t, d] = last_val
        return imputed

    def __repr__(self) -> str:
        return "LOCFImputer()"


class LinearInterpImputer:
    """
    Linear Interpolation along the time axis.

    For sequences of missing values, linearly interpolate between the
    nearest observed values before and after. If no value exists on one
    side, fall back to mean.
    """

    def __init__(self):
        self.means = None

    def fit(self, data: np.ndarray, mask: np.ndarray) -> "LinearInterpImputer":
        D = data.shape[-1]
        self.means = np.zeros(D, dtype=np.float32)
        for d in range(D):
            observed = data[..., d][mask[..., d] == 1]
            if len(observed) > 0:
                self.means[d] = observed.mean()
        return self

    def impute(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if data.ndim == 2:
            return self._impute_single(data, mask)

        imputed = np.zeros_like(data)
        for i in range(data.shape[0]):
            imputed[i] = self._impute_single(data[i], mask[i])
        return imputed

    def _impute_single(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Linear interpolation for a single sample (T, D)."""
        T, D = data.shape
        imputed = data.copy()

        for d in range(D):
            # Find observed indices
            obs_idx = np.where(mask[:, d] == 1)[0]

            if len(obs_idx) == 0:
                # No observations: fill with mean
                imputed[:, d] = self.means[d]
                continue

            if len(obs_idx) == T:
                # All observed, nothing to do
                continue

            # Use numpy interp for linear interpolation
            obs_vals = data[obs_idx, d]
            all_idx = np.arange(T)
            imputed[:, d] = np.interp(all_idx, obs_idx, obs_vals)

        return imputed

    def __repr__(self) -> str:
        return "LinearInterpImputer()"
