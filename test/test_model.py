# File: test_model.py
# File Created: Saturday, 23rd February 2019 8:30:36 pm
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Tests for model.py
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch as th

from gptorch.model import Param, Model, GPModel
from gptorch.kernels import Rbf
from gptorch.models import GPR


class TestParam(object):
    """
    Tests for the Param class
    """
    def test_init(self):
        # Test various permitted inits:
        Param(th.DoubleTensor([1.0]))
        Param(th.DoubleTensor([1.0]), requires_grad=False)
        Param(th.DoubleTensor([1.0]), requires_transform=False)
        Param(th.DoubleTensor([1.0]), requires_grad=False, 
            requires_transform=False)

    def test_access(self):
        """
        Test accessing the value.
        """
        p = Param(th.DoubleTensor([1.0]))
        assert isinstance(p.data, th.DoubleTensor)
        assert isinstance(p.data.numpy(), np.ndarray)

    def test_transform(self):
        """
        Test that parameters requiring a transform return the correct value.

        Currently, we obtain the untransformed variable by default.  Perhaps we
        should switch this in the future.
        """
        p = Param(th.DoubleTensor([1.0]))
        assert p.data.numpy()[0] == 1.0
        pt = Param(th.DoubleTensor([1.0]), requires_transform=True)
        assert p.transform().data.numpy()[0] == 1.0


class TestModel(object):
    """
    Tests for the Model class
    """
    pass


class TestGPModel(object):
    """
    Tests for the GPModel class
    """
    def test_predict_f_samples(self):
        # TODO mock a GPModel?  Using GPR for the moment.
        n, dx, dy = 5, 3, 2
        x, y = np.random.randn(n, dx), np.random.randn(n, dy)
        kern = Rbf(dx, ARD=True)
        gp = GPR(y, x, kern)

        n_test = 5
        x_test = np.random.randn(n_test, dx)
        f_samples = gp.predict_f_samples(x_test)
        assert isinstance(f_samples, th.Tensor)
        assert f_samples.ndimension() == 3  # [sample x n_test x dy]
        assert f_samples.shape == (1, n_test, dy)

        n_samples = 3
        f_samples_2 = gp.predict_f_samples(x_test, n_samples=n_samples)
        assert isinstance(f_samples_2, th.Tensor)
        assert f_samples_2.ndimension() == 3  # [sample x n_test x dy]
        assert f_samples_2.shape == (n_samples, n_test, dy)

    def test_predict_y_samples(self):
        # TODO mock a GPModel?  Using GPR for the moment.
        n, dx, dy = 5, 3, 2
        x, y = np.random.randn(n, dx), np.random.randn(n, dy)
        kern = Rbf(dx, ARD=True)
        gp = GPR(y, x, kern)

        n_test = 5
        x_test = np.random.randn(n_test, dx)
        y_samples = gp.predict_y_samples(x_test)
        assert isinstance(y_samples, th.Tensor)
        assert y_samples.ndimension() == 3  # [sample x n_test x dy]
        assert y_samples.shape == (1, n_test, dy)

        n_samples = 3
        y_samples_2 = gp.predict_y_samples(x_test, n_samples=n_samples)
        assert isinstance(y_samples_2, th.Tensor)
        assert y_samples_2.ndimension() == 3  # [sample x n_test x dy]
        assert y_samples_2.shape == (n_samples, n_test, dy)