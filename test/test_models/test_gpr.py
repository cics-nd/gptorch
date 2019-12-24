# File: test_gpr.py
# File Created: Saturday, 13th July 2019 3:25:43 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import os
import sys

import pytest
import numpy as np
import torch

from gptorch.models import GPR
from gptorch.kernels import Rbf
from gptorch.util import torch_dtype, TensorType

base_path = os.path.join(os.path.dirname(__file__), "..", "..")
if not base_path in sys.path:
    sys.path.append(base_path)

from test.util import needs_cuda


class TestGPR(object):
    def test_init(self):
        n, dx, dy = 5, 3, 2
        x, y = np.random.randn(n, dx), np.random.randn(n, dy)
        kern = Rbf(x.shape[1], ARD=True)

        # init w/ numpy
        GPR(x, y, kern)
        # init w/ PyTorch tensors:
        GPR(TensorType(y), TensorType(x), kern)
        # init w/ a mean function:
        GPR(x, y, kern, mean_function=torch.nn.Linear(dx, dy))

    def test_loss(self):
        model, x, y = self._get_model()
        n = x.shape[0]

        loss = model.loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.ndimension() == 1  # TODO change this...

        # Test ability to specify x and y
        loss_xy = model.loss(x=TensorType(x), y=TensorType(y))
        assert isinstance(loss_xy, torch.Tensor)
        assert loss_xy.item() == loss.item()

        with pytest.raises(ValueError):
            # Size mismatch
            model.loss(x=TensorType(x[:n // 2]))

    @needs_cuda
    def test_compute_loss_cuda(self):
        model = self._get_model()[0]
        model.cuda()

        loss = model.loss()
        assert loss.is_cuda

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

    @needs_cuda
    def test_predict_cuda(self):
        n, n_test, dx, dy = 5, 7, 3, 2
        x, y = torch.randn(n, dx), torch.randn(n, dy)
        kern = Rbf(x.shape[1], ARD=True)
        model = GPR(x, y, kern)
        model.cuda()

        x_test = torch.randn(n_test, dx, dtype=torch_dtype).cuda()
        for t in model._predict(x_test):  # mean, variance
            assert t.is_cuda

    @staticmethod
    def _get_model():
        n, dx, dy = 5, 3, 2
        x, y = np.random.randn(n, dx), np.random.randn(n, dy)
        kern = Rbf(x.shape[1], ARD=True)

        return GPR(x, y, kern), x, y
