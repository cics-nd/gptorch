# Steven Atkinson
# satkinso@nd.edu
# 

"""
Tests on densities:
"""

from __future__ import absolute_import
from gptorch.densities import GaussianMultivariateDiagonal, \
    GaussianMultivariateFull
from gptorch import util
from torch.autograd import Variable
import torch
import numpy as np
from unittest import TestCase
import os

# For computations that are limited by computational precision
comp_tol = 1.0e-12
# For comptutaions that are limited by Monte Carlo convergence (sampling tests)
# This is always dicey, so the tolerance is pretty relaxed
mc_tol = 0.2


def _get_matrix(s):
    x_np = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 
        'densities', s + '.dat'))
    if x_np.size == 1:
        x_np = x_np.reshape((1))
    return util.as_tensor(x_np)


def _err(measured, target, relative=True):
    err = torch.abs(measured - target)
    if relative:
        err /= target
    return torch.max(err).data.numpy()


class TestGaussianMultivariateDiagonal(TestCase):
    @staticmethod
    def _get_matrix(s):
        return _get_matrix(os.path.join('MultivariateGaussianDiagonal/', s))

    @staticmethod
    def _get_mu_sigma():
        return TestGaussianMultivariateDiagonal._get_matrix('mu'), \
               TestGaussianMultivariateDiagonal._get_matrix('sigma')

    def test_pdf(self):
        mu, sigma = TestGaussianMultivariateDiagonal._get_mu_sigma()
        x = TestGaussianMultivariateDiagonal._get_matrix('x')
        pdf_true = TestGaussianMultivariateDiagonal._get_matrix('p_of_x')
        d = GaussianMultivariateDiagonal(mu, sigma)
        pdf_computed = d.pdf(x)
        assert isinstance(pdf_computed, Variable)
        assert len(pdf_computed.size()) == 1  # Return a 1D array
        err = _err(pdf_computed, pdf_true)
        assert err < comp_tol

    def test_sample(self):
        torch.manual_seed(42)  # To ensure repeatability
        mu, sigma = TestGaussianMultivariateDiagonal._get_mu_sigma()
        d = GaussianMultivariateDiagonal(mu, sigma)
        s = d.sample(10 ** 7)
        mu_measured = torch.mean(s, 0)
        cov_measured = util.as_tensor(np.cov(s.data.numpy().transpose()))
        mu_err = _err(mu_measured, mu)
        cov_err = _err(cov_measured, sigma.diag(), False)
        print(mu_err)
        print(cov_err)
        assert mu_err < mc_tol
        assert cov_err < mc_tol


class TestGaussianMultivariateFull(TestCase):
    @staticmethod
    def _get_matrix(s):
        return _get_matrix('MultivariateGaussianFull/' + s)

    @staticmethod
    def _get_mu_sigma():
        return TestGaussianMultivariateFull._get_matrix('mu'), \
               TestGaussianMultivariateFull._get_matrix('sigma')

    def test__get_precision(self):
        mu, sigma = TestGaussianMultivariateFull._get_mu_sigma()
        sigma_inv = TestGaussianMultivariateFull._get_matrix('sigma_inv')
        d = GaussianMultivariateFull(mu, sigma, compute_precision=True)
        precision_err = _err(d._precision, sigma_inv)
        assert precision_err < comp_tol

    def test__get_chol(self):
        mu, sigma = TestGaussianMultivariateFull._get_mu_sigma()
        sigma_chol = TestGaussianMultivariateFull._get_matrix('sigma_cholesky')
        d = GaussianMultivariateFull(mu, sigma, compute_chol=True)
        chol_err = _err(d._l, sigma_chol, False)
        assert chol_err < comp_tol

    def test__get_coefficient(self):
        mu, sigma = TestGaussianMultivariateFull._get_mu_sigma()
        d = GaussianMultivariateFull(mu, sigma)
        d._get_coefficient()
        coefficient_true = TestGaussianMultivariateFull._get_matrix(
            'coefficient')
        coeff_err = _err(d._coefficient, coefficient_true)
        assert coeff_err < comp_tol

    def test_pdf(self):
        mu, sigma = TestGaussianMultivariateFull._get_mu_sigma()
        x = TestGaussianMultivariateFull._get_matrix('x')
        pdf_true = TestGaussianMultivariateFull._get_matrix('p_of_x')
        d = GaussianMultivariateFull(mu, sigma)
        pdf = d.pdf(x)
        pdf_err = _err(pdf, pdf_true)
        assert pdf_err < comp_tol

    def test_sample(self):
        torch.manual_seed(42)  # To ensure repeatability
        mu, sigma = TestGaussianMultivariateFull._get_mu_sigma()
        d = GaussianMultivariateFull(mu, sigma)
        s = d.sample(10 ** 7)
        mu_measured = torch.mean(s, 0)
        cov_measured = util.as_tensor(np.cov(s.data.numpy().transpose()))
        mu_err = _err(mu_measured, mu)
        cov_err = _err(cov_measured, sigma)
        print(mu_err)
        print(cov_err)
        assert mu_err < mc_tol
        assert cov_err < mc_tol
