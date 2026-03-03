"""
Tests for imputation metrics.
"""

import numpy as np
import pytest
import torch

from metrics.imputation import mae, mae_torch, mse_torch, per_variable_metrics, rmse


class TestMAE:
    def test_perfect_prediction(self):
        pred = np.array([[[1, 2], [3, 4]]], dtype=np.float32)  # (1, 2, 2)
        target = pred.copy()
        mask = np.ones_like(pred)

        assert mae(pred, target, mask) == pytest.approx(0.0)

    def test_known_error(self):
        pred = np.array([[[1, 2]]], dtype=np.float32)    # (1, 1, 2)
        target = np.array([[[3, 5]]], dtype=np.float32)  # errors: 2, 3
        mask = np.ones_like(pred)

        assert mae(pred, target, mask) == pytest.approx(2.5)

    def test_masked_positions_only(self):
        pred = np.array([[[1, 100]]], dtype=np.float32)
        target = np.array([[[3, 999]]], dtype=np.float32)
        mask = np.array([[[1, 0]]], dtype=np.float32)  # only evaluate first feature

        assert mae(pred, target, mask) == pytest.approx(2.0)

    def test_empty_mask(self):
        pred = np.zeros((2, 3, 4), dtype=np.float32)
        target = np.ones((2, 3, 4), dtype=np.float32)
        mask = np.zeros((2, 3, 4), dtype=np.float32)

        assert mae(pred, target, mask) == pytest.approx(0.0)


class TestRMSE:
    def test_perfect_prediction(self):
        pred = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
        target = pred.copy()
        mask = np.ones_like(pred)

        assert rmse(pred, target, mask) == pytest.approx(0.0)

    def test_known_error(self):
        pred = np.array([[[0]]], dtype=np.float32)
        target = np.array([[[3]]], dtype=np.float32)
        mask = np.ones_like(pred)

        assert rmse(pred, target, mask) == pytest.approx(3.0)


class TestPerVariableMetrics:
    def test_separate_features(self):
        pred = np.array([[[1, 10]]], dtype=np.float32)
        target = np.array([[[2, 20]]], dtype=np.float32)
        mask = np.array([[[1, 1]]], dtype=np.float32)

        result = per_variable_metrics(pred, target, mask, ["feat_a", "feat_b"])

        assert result["feat_a"]["mae"] == pytest.approx(1.0)
        assert result["feat_b"]["mae"] == pytest.approx(10.0)


class TestTorchMetrics:
    def test_mae_torch_gradient(self):
        pred = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
        target = torch.tensor([[[3.0, 5.0]]])
        mask = torch.tensor([[[1.0, 1.0]]])

        loss = mae_torch(pred, target, mask)
        loss.backward()

        assert pred.grad is not None
        assert loss.item() == pytest.approx(2.5)

    def test_mse_torch(self):
        pred = torch.tensor([[[0.0]]])
        target = torch.tensor([[[3.0]]])
        mask = torch.tensor([[[1.0]]])

        loss = mse_torch(pred, target, mask)
        assert loss.item() == pytest.approx(9.0)
