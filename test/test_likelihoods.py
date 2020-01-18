# File: test_likelihoods.py
# File Created: Wednesday, 13th February 2019 9:33:40 pm
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Tests for likelihood classes
"""

import torch
from torch import distributions
import pytest

from gptorch import likelihoods
import gptorch.model
from gptorch.util import TensorType, torch_dtype, LazyMultivariateNormal


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

    def test_forward(self):
        """
        Ensure that everything gets dispatched and evaluates
        """

        lik = self._standard_likelihood()
        lik(TensorType([0.4]))
        lik(distributions.Normal(TensorType([0.8]), TensorType([1.2])))
        lik(LazyMultivariateNormal(TensorType([[1.2, 3.0]]), torch.eye(2)))

    def test_forward_tensor(self):
        lik = self._standard_likelihood()
        f = TensorType([1.2, 0.8])
        with torch.no_grad():
            py = lik._forward_tensor(f)

            assert isinstance(py, distributions.Normal)
            assert py.loc.allclose(f)
            assert all(
                [vi.numpy() == lik.variance.transform().numpy() for vi in py.variance]
            )

    def test_forward_normal(self):
        lik = self._standard_likelihood()
        mean_f, var_f = TensorType([1.2, 0.8]), TensorType([0.2, 0.3])
        scale_f = var_f.sqrt()
        pf = distributions.Normal(mean_f, scale_f)
        with torch.no_grad():
            py = lik._forward_normal(pf)

            assert isinstance(py, distributions.Normal)
            assert py.loc.allclose(mean_f)

            expected_variance = var_f + lik.variance.transform()
            assert py.variance.allclose(expected_variance)

    def test_forward_multivariate_normal(self):
        lik = self._standard_likelihood()
        mean_f = TensorType([[1.2, 0.8]])
        cov_f = TensorType([0.2, 0.3]).diag()[None, :, :]
        pf = LazyMultivariateNormal(mean_f, cov_f)
        with torch.no_grad():
            py = lik._forward_multivariate_normal(pf)

            assert isinstance(py, distributions.MultivariateNormal)
            assert py.loc.allclose(mean_f)

            expected_cov = cov_f + lik.variance.transform() * torch.eye(
                2, 2, dtype=torch_dtype
            )
            assert py.covariance_matrix.allclose(expected_cov)

    def test_log_marginal_likelihood(self):
        """
        Log-density
        """
        lik = self._standard_likelihood()
        mean = TensorType([0.0])
        target = TensorType([0.1])
        expected_lml = 0.8836465597893728

        # API
        lml = lik.log_marginal_likelihood(mean, target)
        assert isinstance(lml, TensorType)

        # Value
        assert lml.detach().numpy() == pytest.approx(expected_lml)

        # perhaps check against other .forward()s?

    def test_marginal_log_likelihood(self):
        lik = self._standard_likelihood()
        pf = distributions.Normal(
            TensorType([[-0.2], [1.0]]), TensorType([[0.1], [0.2]])
        )
        targets = TensorType([[-0.1], [0.9]])
        expected_mll = -0.73270688  # Snapshot 2020-01-18

        # API
        mll = lik.marginal_log_likelihood(pf, targets)
        assert isinstance(mll, TensorType)

        # Value
        assert mll.detach().numpy() == pytest.approx(expected_mll)

    @property
    def _expected_likelihood_variance(self):
        return 0.01

    def _standard_likelihood(self):
        return likelihoods.Gaussian(variance=self._expected_likelihood_variance)
