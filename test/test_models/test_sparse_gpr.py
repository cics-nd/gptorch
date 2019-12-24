# File: test_sparse_gpr.py
# File Created: Sunday, 1st September 2019 9:14:58 am
# Author: Steven Atkinson (steven@atkinson.mn)

import os
import sys

import pytest
import numpy as np
import torch

from gptorch.models.sparse_gpr import VFE, SVGP
from gptorch.kernels import Matern32
from gptorch import likelihoods
from gptorch import mean_functions
from gptorch.util import torch_dtype, TensorType

from .common import gaussian_predictions

_data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "models", 
    "sparse_gpr")

base_path = os.path.join(os.path.dirname(__file__), "..", "..")
if not base_path in sys.path:
    sys.path.append(base_path)

from test.util import needs_cuda


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


def _get_matrix(name):
    return np.loadtxt(os.path.join(_data_dir, name + ".dat"))


class _InducingData(object):
    """
    A few pieces in common with these models
    """

    @staticmethod
    @atleast_col
    def _xy():
        return _get_matrix("x"), _get_matrix("y")
    
    @staticmethod
    @atleast_col
    def _x_test():
        return _get_matrix("x_test")
    
    @staticmethod
    @atleast_col
    def _z():
        return _get_matrix("z")


class TestVFE(_InducingData):
    def test_init(self):
        x, y = _InducingData._xy()
        kernel = Matern32(x.shape[1], ARD=True)
        VFE(x, y, kernel)
        VFE(x, y, kernel, inducing_points=_InducingData._z())

        # TODO mean

    def test_compute_loss(self):
        x, y = _InducingData._xy()
        z = _InducingData._z()
        kernel = Matern32(1)
        kernel.length_scales.data = torch.zeros(1, dtype=torch_dtype)
        kernel.variance.data = torch.zeros(1, dtype=torch_dtype)
        likelihood = likelihoods.Gaussian(variance=1.0)

        model = VFE(x, y, kernel, inducing_points=z, likelihood=likelihood,
            mean_function=mean_functions.Zero(1))
        loss = model.loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.ndimension() == 0
        # Computed while I trust the result.
        assert loss.item() == pytest.approx(8.842242323920674)

        # Test ability to specify x and y
        loss_xy = model.loss(x=TensorType(x), y=TensorType(y))
        assert isinstance(loss_xy, torch.Tensor)
        assert loss_xy.item() == loss.item()

        with pytest.raises(ValueError):
            # Size mismatch
            model.loss(x=TensorType(x[:x.shape[0] // 2]))

    @needs_cuda
    def test_compute_loss_cuda(self):
        model = self._get_model()
        model.cuda()
        loss = model.loss()
        assert loss.is_cuda

    def test_predict(self):
        """
        Just the ._predict() method
        """

        x, y = _InducingData._xy()
        z = _InducingData._z()
        kernel = Matern32(1)
        kernel.length_scales.data = torch.zeros(1, dtype=torch_dtype)
        kernel.variance.data = torch.zeros(1, dtype=torch_dtype)
        likelihood = likelihoods.Gaussian(variance=1.0)

        model = VFE(x, y, kernel, inducing_points=z, likelihood=likelihood,
            mean_function=mean_functions.Zero(1))

        x_test = torch.Tensor(_InducingData._x_test())
        mu, s = TestVFE._y_pred()
        gaussian_predictions(model, x_test, mu, s)

    @needs_cuda
    def test_predict_cuda(self):
        model = self._get_model()
        model.cuda()

        x_test = torch.randn(4, model.input_dimension, dtype=torch_dtype).cuda()
        for t in model._predict(x_test):
            assert t.is_cuda

    @staticmethod
    @atleast_col
    def _y_pred():
        return _get_matrix("vfe_y_mean"),  _get_matrix("vfe_y_cov")  

    @staticmethod
    def _get_model():
        x, y = _InducingData._xy()
        z = _InducingData._z()
        kernel = Matern32(1)
        kernel.length_scales.data = torch.zeros(1, dtype=torch_dtype)
        kernel.variance.data = torch.zeros(1, dtype=torch_dtype)
        likelihood = likelihoods.Gaussian(variance=1.0)

        model = VFE(x, y, kernel, inducing_points=z, likelihood=likelihood,
            mean_function=mean_functions.Zero(1))

        return model


class TestSVGP(_InducingData):
    def test_init(self):
        x, y = _InducingData._xy()
        kernel = Matern32(x.shape[1], ARD=True)
        SVGP(x, y, kernel)
        SVGP(x, y, kernel, inducing_points=_InducingData._z())

        SVGP(x, y, kernel, mean_function=mean_functions.Constant(y.shape[1]))
        SVGP(x, y, kernel, mean_function=torch.nn.Linear(x.shape[1], y.shape[1]))

    def test_compute_loss(self):
        x, y = _InducingData._xy()
        z = _InducingData._z()
        u_mu, u_l_s = TestSVGP._induced_outputs()
        kernel = Matern32(1)
        kernel.length_scales.data = torch.zeros(1, dtype=torch_dtype)
        kernel.variance.data = torch.zeros(1, dtype=torch_dtype)
        likelihood = likelihoods.Gaussian(variance=1.0)

        model = SVGP(x, y, kernel, inducing_points=z, likelihood=likelihood,
            mean_function=mean_functions.Zero(1))
        model.induced_output_mean.data = TensorType(u_mu)
        model.induced_output_chol_cov.data = model.induced_output_chol_cov.\
            _transform.inv(TensorType(u_l_s))

        loss = model.loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.ndimension() == 0
        # Computed while I trust the result.
        assert loss.item() == pytest.approx(9.534628739243518)

        # Test ability to specify x and y
        loss_xy = model.loss(x=TensorType(x), y=TensorType(y))
        assert isinstance(loss_xy, torch.Tensor)
        assert loss_xy.item() == loss.item()

        with pytest.raises(ValueError):
            # Size mismatch
            model.loss(x=TensorType(x[:x.shape[0] // 2]), y=TensorType(y))

        model_minibatch = SVGP(x, y, kernel, batch_size=1)
        loss_mb = model_minibatch.loss()
        assert isinstance(loss_mb, torch.Tensor)
        assert loss_mb.ndimension() == 0

        model_full_mb = SVGP(x, y, kernel, inducing_points=z, likelihood=likelihood,
            mean_function=mean_functions.Zero(1), batch_size=x.shape[0])
        model_full_mb.induced_output_mean.data = TensorType(u_mu)
        model_full_mb.induced_output_chol_cov.data = model_full_mb.induced_output_chol_cov.\
            _transform.inv(TensorType(u_l_s))
        loss_full_mb = model_full_mb.loss()
        assert isinstance(loss_full_mb, torch.Tensor)
        assert loss_full_mb.ndimension() == 0
        assert loss_full_mb.item() == pytest.approx(loss.item())

        model.loss(model.X, model.Y)  # Just make sure it works!

    @needs_cuda
    def test_compute_loss_cuda(self):
        model = self._get_model()
        model.cuda()

        loss = model.loss()
        assert loss.is_cuda

    def test_predict(self):
        """
        Just the ._predict() method
        """

        x, y = _InducingData._xy()
        z = _InducingData._z()
        u_mu, u_l_s = TestSVGP._induced_outputs()
        kernel = Matern32(1)
        kernel.length_scales.data = torch.zeros(1, dtype=torch_dtype)
        kernel.variance.data = torch.zeros(1, dtype=torch_dtype)
        likelihood = likelihoods.Gaussian(variance=1.0)

        model = SVGP(x, y, kernel, inducing_points=z, likelihood=likelihood,
            mean_function=mean_functions.Zero(1))
        model.induced_output_mean.data = TensorType(u_mu)
        model.induced_output_chol_cov.data = model.induced_output_chol_cov.\
            _transform.inv(TensorType(u_l_s))

        x_test = TensorType(_InducingData._x_test())
        mu, s = TestSVGP._y_pred()
        gaussian_predictions(model, x_test, mu, s)

    @needs_cuda
    def test_predict_cuda(self):
        model = self._get_model()
        model.cuda()

        x_test = torch.randn(4, model.input_dimension, dtype=torch_dtype).cuda()
        for t in model._predict(x_test):
            assert t.is_cuda

    @staticmethod
    @atleast_col
    def _induced_outputs():
        return _get_matrix("q_mu"), _get_matrix("l_s")

    @staticmethod
    @atleast_col
    def _y_pred():
        return _get_matrix("svgp_y_mean"), _get_matrix("svgp_y_cov")

    @staticmethod
    def _get_model():
        x, y = _InducingData._xy()
        z = _InducingData._z()
        u_mu, u_l_s = TestSVGP._induced_outputs()
        kernel = Matern32(1)
        kernel.length_scales.data = torch.zeros(1, dtype=torch_dtype)
        kernel.variance.data = torch.zeros(1, dtype=torch_dtype)
        likelihood = likelihoods.Gaussian(variance=1.0)

        model = SVGP(x, y, kernel, inducing_points=z, likelihood=likelihood,
            mean_function=mean_functions.Zero(1))
        model.induced_output_mean.data = TensorType(u_mu)
        model.induced_output_chol_cov.data = model.induced_output_chol_cov.\
            _transform.inv(TensorType(u_l_s))
        
        return model
