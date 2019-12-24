# File: test_model.py
# File Created: Saturday, 23rd February 2019 8:30:36 pm
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Tests for model.py
"""

import os
import sys
from functools import partial

import numpy as np
import torch
from torch.distributions import Normal

base_path = os.path.join(os.path.dirname(__file__), "..")
if not base_path in sys.path:
    sys.path.append(base_path)

from gptorch.model import Model
from gptorch.param import Param
from gptorch.util import TensorType, torch_dtype

zeros = partial(torch.zeros, dtype=torch_dtype)
ones = partial(torch.ones, dtype=torch_dtype)


def _standard_normal(*shape):
    return Normal(zeros(*shape), ones(*shape))


class _MockModel(Model):
    """
    Mockup model that has a log-likelihood implemented
    """

    def __init__(self):
        super().__init__()
        self.z = Param(zeros(1), prior=_standard_normal(1))
        self.targets = zeros(1)  # Mock training data

    def log_likelihood(self):
        return _standard_normal(1).log_prob(self.targets).sum()

    def _loss(self):
        return -(self.log_likelihood() + self.log_prior())


class TestModel(object):
    """
    Tests for the Model class
    """

    def test_get_param_array(self):

        model = _MockModel()
        param_array = model._get_param_array()

    def test_set_param_array(self):

        model = _MockModel()
        param_array = model._get_param_array()
        param_array += 1.0
        model._set_parameters(param_array)

    def test_loss_and_grad(self):
        model = _MockModel()
        param_array = model._get_param_array()
        loss, grad = model._loss_and_grad(param_array)

    def test_extract_params(self):

        model = _MockModel()
        params = model.extract_params()

    def test_expand_params(self):

        model = _MockModel()
        params = model.extract_params()
        model.expand_params(params)

    def test_log_likelihood(self):
        model = _MockModel()

        target = zeros(1)
        log_likelihood = model.log_likelihood()

        assert isinstance(log_likelihood, TensorType)
        assert log_likelihood.ndimension() == 0
        # See _MockModel to verify target is 0.0, likelihood is a standard normal.
        assert log_likelihood.item() == _standard_normal(1).log_prob(zeros(1)).sum().item()

    def test_log_prior(self):
        """
        Basic test to make sure that log_prior() works in a basic model: 
        Only one parameter, it's 1D, and it has a prior.
        """

        model = _MockModel()
        
        log_prior = model.log_prior()
        assert isinstance(log_prior, TensorType)
        assert log_prior.ndimension() == 0
        # See _MockModel:
        assert log_prior.item() == _standard_normal(1).log_prob(zeros(1)).sum().item()

    def test_loss(self):
        model = _MockModel()

        loss = model.loss()

        assert isinstance(loss, TensorType)
        assert loss.ndimension() == 0
        assert loss == -(model.log_likelihood() + model.log_prior())
