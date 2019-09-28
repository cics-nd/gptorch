# File: test_gpr.py
# File Created: Saturday, 13th July 2019 3:25:43 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest
import numpy as np
import torch

from gptorch.models import GPR
from gptorch.kernels import Rbf
from gptorch.util import torch_dtype, TensorType

torch.set_default_dtype(torch_dtype)


class TestGPR(object):
    def test_init(self):
        n, dx, dy = 5, 3, 2
        x, y = np.random.randn(n, dx), np.random.randn(n, dy)
        kern = Rbf(x.shape[1], ARD=True)

        # init w/ numpy
        GPR(x, y, kern)
        # init w/ PyTorch tensors:
        GPR(torch.Tensor(y), torch.Tensor(x), kern)
        # init w/ a mean function:
        GPR(x, y, kern, mean_function=torch.nn.Linear(dx, dy))

    def test_compute_loss(self):
        n, dx, dy = 5, 3, 2
        x, y = np.random.randn(n, dx), np.random.randn(n, dy)
        kern = Rbf(x.shape[1], ARD=True)

        model = GPR(x, y, kern)
        loss = model.compute_loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.ndimension() == 1  # TODO change this...

        # Test ability to specify x and y
        loss_xy = model.compute_loss(x=TensorType(x), y=TensorType(y))
        assert isinstance(loss_xy, torch.Tensor)
        assert loss_xy.item() == loss.item()

        with pytest.raises(ValueError):
            # Size mismatch
            model.compute_loss(x=TensorType(x[:n // 2]))

    def test_predict(self):
        n, n_test, dx, dy = 5, 7, 3, 2
        x, y = torch.randn(n, dx), torch.randn(n, dy)
        kern = Rbf(x.shape[1], ARD=True)
        model = GPR(x, y, kern)

        x_test = torch.randn(n_test, dx)
        mu_var, var = model._predict(x_test)
        assert all([e == a for e, a in zip(mu_var.shape, (n_test, dy))])
        assert all([e == a for e, a in zip(var.shape, (n_test, dy))])

        mu_cov, cov = model._predict(x_test, diag=False)
        assert all([e == a for e, a in zip(mu_cov.shape, (n_test, dy))])
        assert all([e == a for e, a in zip(cov.shape, (n_test, n_test))])
