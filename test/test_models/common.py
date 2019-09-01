# File: common.py
# File Created: Sunday, 1st September 2019 9:57:32 am
# Author: Steven Atkinson (steven@atkinson.mn)

import torch
import numpy as np
import pytest

from gptorch.models.base import GPModel
from gptorch.util import torch_dtype

torch.set_default_dtype(torch_dtype)


def gaussian_predictions(model: GPModel, x_test: torch.Tensor, 
        expected_mu: np.ndarray, expected_s: np.ndarray):
    """
    Every GP model with a gaussian likelihood needs the same set of tests run on
    its ._predict() method.
    """

    # Predictions without full covariance
    mu_diag, s_diag = model._predict(x_test, diag=True)

    assert isinstance(mu_diag, torch.Tensor)
    assert isinstance(s_diag, torch.Tensor)

    assert mu_diag.shape[0] == x_test.shape[0]
    assert mu_diag.shape[1] == model.Y.shape[1]
    assert all([ss == ms for ss, ms in zip(mu_diag.shape, s_diag.shape)])

    assert all([a == pytest.approx(e) for a, e in zip(
        mu_diag.detach().numpy().flatten(), expected_mu.flatten())])
    assert all([a == pytest.approx(e) for a, e in zip(
        s_diag.detach().numpy().flatten(), expected_s.diagonal().flatten())])

    # Predictions with full covariance
    mu_full, s_full = model._predict(x_test, diag=False)

    assert isinstance(mu_full, torch.Tensor)
    assert isinstance(s_full, torch.Tensor)

    assert mu_full.shape[0] == x_test.shape[0]
    assert mu_full.shape[1] == model.Y.shape[1]
    assert all([ss == x_test.shape[0] for ss in s_full.shape])
    
    assert all([a == pytest.approx(e) for a, e in zip(
        mu_full.detach().numpy().flatten(), expected_mu.flatten())])
    assert all([a == pytest.approx(e) for a, e in zip(
        s_full.detach().numpy().flatten(), expected_s.flatten())])
