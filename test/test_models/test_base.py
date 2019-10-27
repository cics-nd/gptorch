# File: test_base.py
# File Created: Saturday, 13th July 2019 1:54:18 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest
import torch
import numpy as np

from gptorch.models.base import GPModel
from gptorch.mean_functions import Zero
from gptorch.util import TensorType
from gptorch.kernels import Rbf
from gptorch.models import GPR


class TestGPModel(object):
    """
    Tests for the GPModel class
    """

    def test_cuda(self):
        gp = self._get_model()

        gp.cuda()
        # Ensure that the data made it too
        assert gp.X.is_cuda
        assert gp.Y.is_cuda

    def test_cpu(self):
        gp = self._get_model()

        gp.cuda()
        gp.cpu()

        # Ensure that the data made it too
        assert not gp.X.is_cuda
        assert not gp.Y.is_cuda

    def test_predict_f(self):
        self._predict_fy("predict_f")

    def test_predict_y(self):
        self._predict_fy("predict_y")

    def test_predict_f_samples(self):
        self._predict_fy_samples("predict_f_samples")

    def test_predict_y_samples(self):
        self._predict_fy_samples("predict_y_samples")

    def _predict_fy(self, attr):
        """
        attr='predict_f' or 'predict_y'
        """

        n, dx, dy = 5, 3, 2
        x, y = np.random.randn(n, dx), np.random.randn(n, dy)
        kern = Rbf(dx, ARD=True)
        gp = GPR(x, y, kern)

        n_test = 5
        x_test = np.random.randn(n_test, dx)
        f = getattr(gp, attr)
        mu, v = f(x_test)
        for result in [mu, v]:
            assert isinstance(result, np.ndarray)
            assert result.ndim == 2  # [n_test x dy]
            assert result.shape == (n_test, dy)

        x_test_torch = TensorType(x_test)
        mu_torch, v_torch = f(x_test_torch)
        for result in [mu_torch, v_torch]:
            assert isinstance(result, TensorType)
            assert result.ndimension() == 2  # [n_test x dy]
            assert result.shape == (n_test, dy)

    def _predict_fy_samples(self, attr):
        """
        attr="predict_f_samples" or "predict_y_samples"
        """

        # TODO mock a GPModel?  Using GPR for the moment.
        n, dx, dy = 5, 3, 2
        x, y = np.random.randn(n, dx), np.random.randn(n, dy)
        kern = Rbf(dx, ARD=True)
        gp = GPR(x, y, kern)
        f = getattr(gp, attr)

        n_test = 5
        x_test = np.random.randn(n_test, dx)
        samples = f(x_test)
        assert isinstance(samples, np.ndarray)
        assert samples.ndim == 3  # [sample x n_test x dy]
        assert samples.shape == (1, n_test, dy)

        n_samples = 3
        samples_2 = f(x_test, n_samples=n_samples)
        assert isinstance(samples_2, np.ndarray)
        assert samples_2.ndim == 3  # [sample x n_test x dy]
        assert samples_2.shape == (n_samples, n_test, dy)

        x_test_torch = TensorType(x_test)
        samples_torch = f(x_test_torch)
        assert isinstance(samples_torch, TensorType)
        assert samples_torch.ndimension() == 3  # [1 x n_test x dy]
        assert samples_torch.shape == (1, n_test, dy)

    @staticmethod
    def _get_model():
        n, dx, dy = 5, 3, 2
        x, y = np.random.randn(n, dx), np.random.randn(n, dy)
        kern = Rbf(dx, ARD=True)
        return GPR(x, y, kern)
