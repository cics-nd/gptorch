# File: test_sparse_gpr.py
# File Created: Sunday, 1st September 2019 9:14:58 am
# Author: Steven Atkinson (steven@atkinson.mn)

import os

import pytest
import numpy as np
import torch

from gptorch.models.sparse_gpr import VFE, SVGP
from gptorch.kernels import Matern32
from gptorch import likelihoods
from gptorch import mean_functions
from gptorch.util import torch_dtype

from .common import gaussian_predictions

torch.set_default_dtype(torch_dtype)

_data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "models", 
    "sparse_gpr")


def atleast_col(func):
    """
    Decorator making sure that matrices loaded are column vectors if 1D
    """

    def wrapped():
        outputs = func()
        if isinstance(outputs, tuple):
            outputs = [o[:, np.newaxis] if o.ndim == 1 else o for o in outputs]
        else:
            outputs = outputs[:, np.newaxis] if outputs.ndim == 1 else outputs
        return outputs

    return wrapped


class TestVFE(object):
    def test_init(self):
        x, y = TestVFE._xy()
        kernel = Matern32(x.shape[1], ARD=True)
        VFE(x, y, kernel)
        VFE(x, y, kernel, inducing_points=TestVFE._z())

        # TODO mean

    def test_comptue_loss(self):
        x, y = TestVFE._xy()
        z = TestVFE._z()
        kernel = Matern32(1)
        kernel.length_scales.data = torch.zeros(1)
        kernel.variance.data = torch.zeros(1)
        likelihood = likelihoods.Gaussian(variance=1.0)

        model = VFE(x, y, kernel, inducing_points=z, likelihood=likelihood,
            mean_function=mean_functions.Zero(1))
        loss = model.compute_loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.ndimension() == 0
        # Computed while I trust the result.
        assert loss.item() == pytest.approx(8.842242323920674)

    def test_predict(self):
        """
        Just the ._predict() method (.predict_f, etc tested as part of GPModel)

        TODO Standardize predictions for all GP+Gaussian models
        """

        x, y = TestVFE._xy()
        z = TestVFE._z()
        kernel = Matern32(1)
        kernel.length_scales.data = torch.zeros(1)
        kernel.variance.data = torch.zeros(1)
        likelihood = likelihoods.Gaussian(variance=1.0)

        model = VFE(x, y, kernel, inducing_points=z, likelihood=likelihood,
            mean_function=mean_functions.Zero(1))

        x_test = torch.Tensor(TestVFE._x_test())
        mu, s = TestVFE._y_pred()
        gaussian_predictions(model, x_test, mu, s)

    @staticmethod
    @atleast_col
    def _xy():
        x = TestVFE._get_matrix("x")
        y = TestVFE._get_matrix("y")
        if x.ndim == 1:
            x = x[:, np.newaxis]
        return x, y
    
    @staticmethod
    @atleast_col
    def _x_test():
        return TestVFE._get_matrix("x_test")

    @staticmethod
    @atleast_col
    def _y_pred():
        return TestVFE._get_matrix("y_mean"), TestVFE._get_matrix("y_cov")
    
    @staticmethod
    @atleast_col
    def _z():
        return TestVFE._get_matrix("z")

    @staticmethod
    def _get_matrix(name):
        return np.loadtxt(os.path.join(_data_dir, name + ".dat"))