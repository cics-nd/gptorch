#
# Apr 17, 2017  Yinhao Zhu
#
# class for probability density functions
from __future__ import absolute_import
from . import util as ut
from .functions import cholesky, inverse, cholesky_inverse

import torch as th
import numpy as np
from torch.autograd import Variable
from torch import DoubleTensor
import torch
from scipy.special import factorial
import scipy.stats


class Density(object):
    """
    Superclass for densities
    """
    # def __init__(self):
    #     ...

    def __init__(self, normalization=ut.as_tensor(1.0)):
        self._dim = 0
        # _coefficient ensures that the pdf integrates to 1
        # Change normalization if you want an unnormalized density
        # (integrates to something other than 1)
        self._coefficient = 1.0
        self.normalization = normalization

    def pdf(self, x):
        raise NotImplementedError

    def log_pdf(self, x):
        # Override if needed (e.g. for better numerics)
        return torch.log(self.pdf(x))

    def sample(self, num_samples=1):
        raise NotImplementedError

    def get_dim(self):
        return self._dim

    def get_coefficient(self):
        return self._coefficient


class Exponential(Density):
    """
    Exponential distribution
    p(x; l) = l * exp(-x/l)

    mean = l
    variance = l^2
    """
    def __init__(self, length_scale):
        """
        Args:
            length_scale (torch.autograd.Variable): row vector of length scales
        """
        super(Exponential, self).__init__()
        if len(length_scale.size()) < 2:
            length_scale = length_scale.view(1, -1)
        self.length_scale = length_scale
        self._coefficient = 1.0 / th.prod(length_scale)
        self._dim = length_scale.size(1)

    def pdf(self, x):
        """
        Args:
            x (torch.autograd.Variable): argument
        Returns:
            (torch.autograd.Variable) the density at x
        """
        return self._coefficient * th.prod(th.exp(-x / self.length_scale))

    def log_pdf(self, x):
        return th.log(self._coefficient) + th.sum(-x / self.length_scale)

    def sample(self, num_samples=1):
        """
        Each row is a sample.

        :param num_samples:
        :return: (Variable)
        """
        return self.length_scale * Variable(DoubleTensor(
            np.random.exponential(size=(num_samples, self._dim))))


class InverseGamma(Density):
    """
    Inverse gamma distribution
    p(x; a, b) = b^(-1) / gamma(a) * x^(-a - 1) * exp(-1 / (b * x))
    a = shape
    b = rate (aka 1 / scale)

    Reference: https://en.wikipedia.org/wiki/Inverse-gamma_distribution

    mean = 1 / (b * (a - 1)) if a > 1
    mode = 1 / (b * (a + 1))
    variance = 1 / (b^2 * (a-1)^2 * (a-2)) if a > 2
    """

    def __init__(self, shape, rate, mean=None, variance=None):
        """
        Args:
            shape (torch.autograd.Variable): row vector of shape parameters
            rate (torch.autograd.Variable): row vector of rate parameters
        """
        super(InverseGamma, self).__init__()
        if mean is not None:
            assert variance is not None, "Must provide variance with mean"
            assert isinstance(mean, Variable), \
                "Provide mean as Pytorch Variable"
            assert isinstance(variance, Variable), \
                "Provide variance as Pytorch Variable"
            # Compute shape & rate from provided mean & var
            shape = th.pow(mean, 2) / variance + 2.0
            rate = th.pow(mean * (shape - 1.0), -1)
        assert isinstance(shape, Variable), "Provide shape as torch Variable"
        assert isinstance(rate, Variable), "Provide rate as torch Variable"
        if len(shape.size()) < 2:  # Given as 1d tensor
            shape = shape.view(1, -1)
        if len(rate.size()) < 2:  # Given as 1d tensor
            rate = rate.view(1, -1)
        self.shape = shape
        self.rate = rate
        self._coefficient = th.pow(rate, -shape) / \
                            Variable(th.exp(th.lgamma(shape.data)))
        self._dim = shape.size(1)

        # For RNG using gamma distribution:
        self._gamma_shape = self.shape.data.numpy().flatten()
        self._gamma_scale = self.rate.data.numpy().flatten()

    def pdf(self, x):
        """
        Note: not sure about scipy implementation, so coding up my own.

        Args:
            x (torch.autograd.Variable): argument
        Returns:
            (torch.autograd.Variable) the density at x
        """
        return th.prod(self._coefficient * th.pow(x, -self.shape - 1.0) * \
                       th.exp(-1.0 / (self.rate * x)))

    def sample(self, num_samples=1):
        """
        Each row is a sample.

        Strategy: use numpy's gamma-distributed rng, then transform

        :param num_samples:
        :return: (Variable)
        """
        return ut.as_tensor(np.random.gamma(shape=self._gamma_shape,
                                              scale=self._gamma_scale,
                                              size=(num_samples, self._dim))
                              ** (-1))


class UniformCube(Density):
    def __init__(self, lower_bounds, upper_bounds):
        """
        Args:
            lower_bounds (torch.autograd.Variable): row vector of lower bounds
            upper_bounds (torch.autograd.Variable): row vector of upper bounds
        """
        super(UniformCube, self).__init__()
        if len(lower_bounds.size()) < 2:
            lower_bounds = lower_bounds.view(1, -1)
        if len(upper_bounds.size()) < 2:
            upper_bounds = upper_bounds.view(1, -1)

        assert lower_bounds.size(1) == upper_bounds.size(1), \
            "Sizes of lower and upper bounds do not match"

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.range = upper_bounds - lower_bounds
        self._coefficient = 1.0 / th.prod(self.range)
        self._dim = lower_bounds.size(1)

    def pdf(self, x):
        """
        Args:
            x (torch.autograd.Variable): argument
        Returns:
            (torch.autograd.Variable, 1D) the density at x (
        """
        return self._coefficient * \
            torch.prod((x < self.upper_bounds) * (x >= self.lower_bounds), 1).\
            type(DoubleTensor)

    def sample(self, num_samples=1):
        """
        Each row is a sample.

        :param num_samples:
        :return: (Variable)
        """
        return self.lower_bounds + self.range * \
            ut.as_tensor(np.random.rand(num_samples, self._dim))


# TODO
class GaussianUnivariate(Density):
    """
        Multivariate Gaussian distribution with diagonal covariance matrix
        p(x; m, s) = (2*pi)^(-d/2) |s|^(-1/2) *
                     exp(-1/2 sum_{i=1}^d ((x_i - m_i) / s_i)^2
        mean m and vector of standard deviations s

        mean = mu
        variance = sigma_var = self.sigma^2
        """

    def __init__(self, mu, sigma_var, normalization=ut.as_tensor(1.0)):
        """
        Args:
            mu (torch.autograd.Variable): vector of means
            sigma_var (torch.autograd.Variable): vector of variances
        """
        super(GaussianMultivariateDiagonal, self).__init__(normalization)
        assert mu.numel() == sigma_var.numel(), \
            "Mean and variance vectors must be the same size"

        mu = mu.view(-1)
        sigma_std = torch.sqrt(sigma_var).view(-1)
        self.mu = mu
        self.sigma_std = sigma_std
        self._dim = mu.numel()

    def pdf(self, x):
        """
        Args:
            x (torch.autograd.Variable): argument.  Must be 2-dimensional
                matrix.
        Returns:
            (torch.autograd.Variable) the density at x
        """

        return self.normalization * ut.as_tensor(np.array([np.prod(
            scipy.stats.norm.pdf(
                x_i.data.numpy(),
                self.mu.data.numpy(),
                self.sigma_std.data.numpy())
        ) for x_i in x]))

    def sample(self, num_samples=1):
        """
        Each row is a sample.

        :param num_samples:
        :return: (Variable)
        """
        return ut.as_tensor(np.random.normal(
            size=(num_samples, self._dim),
            loc=self.mu.data.numpy().reshape((1, -1)),
            scale=self.sigma_std.data.numpy().reshape((1, -1))))


class GaussianMultivariateDiagonal(Density):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    p(x; m, s) = (2*pi)^(-d/2) |s|^(-1/2) *
                 exp(-1/2 sum_{i=1}^d ((x_i - m_i) / s_i)^2
    mean m and vector of standard deviations s

    mean = mu
    variance = sigma_var = self.sigma^2
    """
    def __init__(self, mu, sigma_var, normalization=ut.as_tensor(1.0)):
        """
        Args:
            mu (torch.autograd.Variable): vector of means
            sigma_var (torch.autograd.Variable): vector of variances
        """
        super(GaussianMultivariateDiagonal, self).__init__(normalization)
        assert mu.numel() == sigma_var.numel(), \
            "Mean and variance vectors must be the same size"

        mu = mu.view(-1)
        sigma_std = torch.sqrt(sigma_var).view(-1)
        self.mu = mu
        self.sigma_std = sigma_std
        self._dim = mu.numel()

    def pdf(self, x):
        """
        Args:
            x (torch.autograd.Variable): argument.  Must be 2-dimensional
                matrix.
        Returns:
            (torch.autograd.Variable) the density at x
        """

        return self.normalization * ut.as_tensor(np.array([np.prod(
            scipy.stats.norm.pdf(
                x_i.data.numpy(),
                self.mu.data.numpy(),
                self.sigma_std.data.numpy())
            ) for x_i in x]))

    def sample(self, num_samples=1):
        """
        Each row is a sample.

        :param num_samples:
        :return: (Variable)
        """
        return ut.as_tensor(np.random.normal(
            size=(num_samples, self._dim),
            loc=self.mu.data.numpy().reshape((1, -1)),
            scale=self.sigma_std.data.numpy().reshape((1, -1))))


class GaussianMultivariateFull(Density):
    """
    Multivariate Gaussian distribution with full covariance matrix
    p(x; m, S) = (2*pi)^(-d/2) |S|^(-1/2) *
                 exp(-1/2 (x-m)^T S^(-1) (x-m))
    mean m
    covariance matrix S

    mean = mu
    variance = sigma_var = self.sigma^2
    """
    def __init__(self, mu, sigma, normalization=ut.as_tensor(1.0),
                 compute_precision=False, compute_chol=False):
        """
        Args:
            mu (torch.autograd.Variable): vector of means
            sigma_var (torch.autograd.Variable): vector of variances
        """
        super(GaussianMultivariateFull, self).__init__(normalization)
        assert mu.numel() == sigma.size(0), \
            "Mean and variance vectors must be the same size"

        self.mu = mu.view(-1)
        self.sigma = sigma
        self._l = None  # Upper Cholesky of sigma
        self._precision = None
        self._coefficient = None
        self._dim = mu.numel()
        if compute_precision:
            self._get_precision()
        if compute_chol:
            self._get_chol()

    def _get_precision(self):
        if self._l is None:
            self._get_chol()
        self._precision = Variable(cholesky_inverse(self._l.data.t()))

    def _get_chol(self):
        self._l = cholesky(self.sigma).t()

    def _get_coefficient(self):
        if self._l is None:
            self._get_chol()
        self._coefficient = (2.0 * np.pi) ** (-0.5 * self._dim) / \
                            torch.prod(self._l.diag())

    def pdf(self, x):
        """
        Args:
            x (torch.autograd.Variable): argument
        Returns:
            (torch.autograd.Variable) the density at x
        """
        if self._precision is None:
            self._get_precision()
        if self._coefficient is None:
           self._get_coefficient()

        return self.normalization * self._coefficient * torch.cat([
            torch.exp(-0.5 * torch.mm((xi - self.mu).view(1, -1),
                                      torch.mm(self._precision,
                                               (xi - self.mu).view(-1, 1))))
            for xi in x.split(1)]).view(-1)

    def sample(self, num_samples=1):
        """
        Each row is a sample.
        Stores the eigendecomposition of the covariance matrix so that
        subsequent sampling is very fast

        :param num_samples:
        :return: (Variable)
        """
        if self._l is None:
            self._get_chol()
        # mu + eval * randn(n, d) * evec
        return self.mu.view(1, -1) + torch.mm(
            Variable(torch.normal(torch.zeros(num_samples, self._dim),
                                  torch.ones(num_samples, self._dim)).
                type(DoubleTensor)),
            self._l)


def gaussianMD(x, mu, covar):
    """Calculates multi-dimension gaussian density
    Ref: PRML (Bishop) pg. 25

    TODO: make work for 1D also and delete gassian1D, also look into
    torch.normal

    Args:
        x (th.FloatTensor) = 1xD column tensor of the point of interest
        mu (th.FloatTensor) = 1xD column tensor of means 
        covar (th.FloatTensor) = DxD covariance matrix 
    """

    D = x.size()[0] #Dimension of gaussian
    if(D == 1):
        print('1D matrix detected, please use 1D gaussian method')
        return -1

    e = th.mm(th.transpose((x-mu),0,1), th.inverse(covar))
    e = th.mm(e, (x-mu))
    e = th.exp(-0.5*e)
    
    covar0 = Variable(covar).data.numpy()
    det = np.linalg.det(covar0)
    return 1.0/np.power(2*np.pi,D/2.0) * 1.0/(np.power(det,0.5)) * e


def gaussian1D(x, mu, var):
    """Calculates 1D gaussian density
    Ref: PRML (Bishop) pg. 24

    Args:
        x (Variable) = point of interest
        mu (Variable) = mean 
        var (Variable) = Variance squared 
    """
    e = (x-mu)*(1/var)
    e = e*(x-mu)
    e = np.exp(-0.5*e)

    return 1.0/(np.power(2*np.pi*var,0.5))*e


def gamma1D(x, a0, b0):
    """Calculates multi-dimension gamma density
    Ref: PRML (Bishop) pg. 100

    Args:
        x (Variable) = point of interest
        a0 (Variable) = gamma parameter 1 
        b0 (Variable) =  gamma parameter 2
    """
    n = 1/np.exp(ut.gammaln(a0))
    return  n * np.power(b0,a0) * np.power(x,a0-1) * np.exp(-b0 * x)


def studentT(x, mu, lam, nu):
    """Calculates multi-dimensional student T density
    Ref: PRML (Bishop) pg. 105

    **NOTE: it appears for large nu over 1e5 floating point precision starts to fail!**

    Args:
        x (th.FloatTensor) = 1xD column tensor of the point of interest
        mu (th.FloatTensor) = 1xD column tensor of means 
        lam (th.FloatTensor) = DxD precision matrix (inverse of covar)
        nu (Variable) = degrees of freedom
    """
    D = x.size()[0] #Dimension

    #Calculate constant out front
    c0 = np.exp(ut.gammaln((D + nu)/2.0) - ut.gammaln(nu/2.0))
    c0 = c0 * np.power(np.linalg.det(Variable(lam).data.numpy()),0.5) / np.power(np.pi*nu,D/2.0)
    #Mahalanobis Distance
    mah = th.mm(th.transpose((x-mu),0,1), lam)
    mah = th.mm(mah,(x-mu))
    #Tensor calculations
    t0 = th.add(th.div(mah,nu),1)
    t0 = th.pow(t0,(-(D+nu)/2.0))

    return th.mul(t0,c0)


def gaussian(x, mu, var):
    return float(-0.5 * np.log(2 * np.pi)) - 0.5 * th.log(var) \
           - 0.5 * th.pow(x - mu, 2) / var

