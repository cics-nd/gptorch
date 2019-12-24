
"""
Likelihood classes

Objects that propagate either points of distributions through a transformation
yielding a (likelihood) distribution to be evaluated at observed targets 
(outputs) to guide Bayesian inference.

We also desire to be able to compute expectations of log-densities marginalizing 
over input distributions (i.e. "marginal log-likelihoods") as typically occur in
variational inference.
"""

import abc
from math import pi

from torch.nn import Parameter
import torch
from torch import distributions

from .model import Model, Param
from .settings import DefaultPositiveTransform
from .util import torch_dtype

from .util import TensorType


class Likelihood(Model):
    """
    Probabilities that conventionally factorize across data.
    Typically used as the "second stage" of a GP model that goes 
    x -(GP)-> f -(likelihood)-> y
    x = inputs
    f = latent outputs
    y = observed outputs

    We typically have two uses for these:
    1) Do the marginalization in p(y|x) = \int p(y|f) p(f|x) df 
        (e.g. predictions)
    2) Do the marginalization in logp(y) >= \int logp(y|f) p(f) df
        (e.g. variational inference)
    """

    def __init__(self):
        super(Likelihood, self).__init__()

    def predict_mean_variance(self, mean_f, var_f):
        """
        The likelihood of observations y given latent functions f is p(y | f)
        It computes the mean and variance of observations Y given latent
        functions F, which itself is Gaussian.

        p(y) = \int p(y | f) p(f) df

        Gauss-Hermite quadrature is used for numerical integration.

        :param mean_f: mean of latent functions
        :type mean_f: TensorType
        :param var_f: variance of the latent functions
        :type var_f: TensorType
        :return: (TensorType, TensorType) mean and variance of the observations
        """
        # TODO: Gauss-Hermite quadrature
        raise NotImplementedError

    def forward(self):
        return None

    @abc.abstractmethod
    def propagate_log(
        self, qf: torch.distributions.Distribution, targets: torch.Tensor
    ):
        """
        Evaluate the marginal log-likelihood at the targets:
        <log p(y|f)>_q(f)
        Used in variational inference.
        """
        raise NotImplementedError("Implement quadrature fallback")


class Gaussian(Likelihood):
    """
    (Spherical) Gaussian likelihood p(y|f)
    """

    def __init__(self, variance=1.0):
        super(Gaussian, self).__init__()
        self.variance = Param(
            TensorType([variance]), transform=DefaultPositiveTransform()
        )

    def logp(self, F, Y):
        """
        Evaluate the log-density of targets at Y, given a Gaussian density 
        centered at F.

        :param F: Center of the density
        :type F: TensorType
        :param Y: Targets where we want to compute the log-pdf
        :type Y: TensorType
        """
        return distributions.Normal(F, torch.sqrt(self.variance.transform())).log_prob(
            Y
        )

    def predict_mean_variance(self, mean_f, var_f):
        """
        Integrate the input (a Gaussian with provided mean & variance) over the
        likelihood density

        :param mean_f: Mean of input Gaussian
        :type mean_f: TensorType
        :param var_f: Variance of input Gaussian
        :type var_f: TensorType

        :return: (TensorType, TensorType) mean & variance
        """
        # TODO: consider mulit-output case
        # stupid syntax - expecting broadcasting in PyTorch
        return mean_f, var_f + self.variance.transform().expand_as(var_f)

    def predict_mean_covariance(self, mean_f, cov_f):
        return mean_f, cov_f + self.variance.transform().expand_as(cov_f).diag().diag()

    def propagate_log(self, qf, targets):
        if not isinstance(qf, torch.distributions.Normal) and not isinstance(
            qf, torch.distributions.MultivariateNormal
        ):
            raise TypeError("Expect Gaussian q(f)")

        mu, s = qf.loc, qf.variance
        n = targets.nelement()
        if not mu.nelement() == n:
            raise ValueError(
                "Targets (%i) and q(f) (%i) have mismatch in size" % (n, mu.nelement())
            )

        sigma_y = self.variance.transform()

        return -0.5 * (
            n * (torch.log(TensorType([2.0 * pi])).to(sigma_y.device) 
            + torch.log(sigma_y))
            + (torch.sum((targets - mu) ** 2) + s.sum()) / sigma_y
        )
