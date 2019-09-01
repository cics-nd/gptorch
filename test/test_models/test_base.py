# File: test_base.py
# File Created: Saturday, 13th July 2019 1:54:18 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest
import torch
import numpy as np

from gptorch.models.base import GPModel
from gptorch.mean_functions import Zero
from gptorch.util import torch_dtype
from gptorch.kernels import Rbf
from gptorch.models import GPR

torch.set_default_dtype(torch_dtype)


class TestGPModel(object):
    """
    Tests for the GPModel class
    """

    def test_predict_f_samples(self):
        # TODO mock a GPModel?  Using GPR for the moment.
        n, dx, dy = 5, 3, 2
        x, y = np.random.randn(n, dx), np.random.randn(n, dy)
        kern = Rbf(dx, ARD=True)
        gp = GPR(x, y, kern)

        n_test = 5
        x_test = np.random.randn(n_test, dx)
        f_samples = gp.predict_f_samples(x_test)
        assert isinstance(f_samples, torch.Tensor)
        assert f_samples.ndimension() == 3  # [sample x n_test x dy]
        assert f_samples.shape == (1, n_test, dy)

        n_samples = 3
        f_samples_2 = gp.predict_f_samples(x_test, n_samples=n_samples)
        assert isinstance(f_samples_2, torch.Tensor)
        assert f_samples_2.ndimension() == 3  # [sample x n_test x dy]
        assert f_samples_2.shape == (n_samples, n_test, dy)

    def test_predict_y_samples(self):
        # TODO mock a GPModel?  Using GPR for the moment.
        n, dx, dy = 5, 3, 2
        x, y = np.random.randn(n, dx), np.random.randn(n, dy)
        kern = Rbf(dx, ARD=True)
        gp = GPR(x, y, kern)

        n_test = 5
        x_test = np.random.randn(n_test, dx)
        y_samples = gp.predict_y_samples(x_test)
        assert isinstance(y_samples, torch.Tensor)
        assert y_samples.ndimension() == 3  # [sample x n_test x dy]
        assert y_samples.shape == (1, n_test, dy)

        n_samples = 3
        y_samples_2 = gp.predict_y_samples(x_test, n_samples=n_samples)
        assert isinstance(y_samples_2, torch.Tensor)
        assert y_samples_2.ndimension() == 3  # [sample x n_test x dy]
        assert y_samples_2.shape == (n_samples, n_test, dy)
