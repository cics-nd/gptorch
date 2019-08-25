# File: test_likelihoods.py
# File Created: Wednesday, 13th February 2019 9:33:40 pm
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Tests for likelihood classes
"""

import torch as th
from torch.autograd import Variable
import pytest

from gptorch import likelihoods
import gptorch.model
from gptorch.util import TensorType


class TestGaussian(object):
    """
    Tests for Gaussian likelihood
    """

    def test_init(self):
        """
        Ensure that initialization succeeds
        """
        likelihoods.Gaussian()
        self._standard_likelihood()

    def test_variance(self):
        """
        Test variance API
        """
        lik = self._standard_likelihood()

        # Type
        assert isinstance(lik.variance, gptorch.model.Param)

        # Value
        print(type(lik.variance.transform()))
        assert lik.variance.transform().data.numpy() == pytest.approx(
            self._expected_likelihood_variance
        )

    def test_logp(self):
        """
        Log-density
        """
        lik = self._standard_likelihood()
        mean = Variable(TensorType([0.0]))
        target = Variable(TensorType([0.1]))
        expected_logp = 0.8836465597893728

        # API
        logp = lik.logp(mean, target)
        assert isinstance(logp, Variable)

        # Value
        assert logp.data.numpy() == pytest.approx(expected_logp)

    def test_predict_mean_variance(self):
        """
        Test propagation of diagonal Gaussian through the likelihood density.
        """
        lik = self._standard_likelihood()
        input_mean = Variable(TensorType([0.0]))
        input_variance = Variable(TensorType([1.0]))
        expected_output_mean = input_mean
        expected_output_variance = input_variance + self._expected_likelihood_variance

        # API
        output_mean, output_variance = lik.predict_mean_variance(
            input_mean, input_variance
        )
        assert isinstance(output_mean, Variable)
        assert isinstance(output_variance, Variable)

        # Value
        assert output_mean.data.numpy() == expected_output_mean.data.numpy()
        assert output_variance.data.numpy() == pytest.approx(
            expected_output_variance.data.numpy()
        )

    def test_predict_mean_covariance(self):
        """
        Test propagation of full Gaussian through the likelihood density.
        """
        lik = self._standard_likelihood()
        input_mean = Variable(TensorType([0.0, 1.0, 2.1]))
        input_covariance = Variable(
            TensorType([[1.0, 0.5, 0.2], [0.5, 1.0, 0.5], [0.2, 0.5, 1.0]])
        )
        expected_output_mean = input_mean
        # Ugh, sorry about this.  Will cleanup when we move PyTorch forward!
        expected_output_covariance = (
            input_covariance
            + Variable(TensorType([self._expected_likelihood_variance]))
            .expand_as(input_covariance)
            .diag()
            .diag()
        )

        # API
        output_mean, output_covariance = lik.predict_mean_covariance(
            input_mean, input_covariance
        )
        assert isinstance(output_mean, Variable)
        assert isinstance(output_covariance, Variable)

        # Value
        assert all(output_mean.data.numpy() == expected_output_mean.data.numpy())
        assert output_covariance.data.numpy() == pytest.approx(
            expected_output_covariance.data.numpy()
        )

    @property
    def _expected_likelihood_variance(self):
        return 0.01

    def _standard_likelihood(self):
        return likelihoods.Gaussian(variance=self._expected_likelihood_variance)
