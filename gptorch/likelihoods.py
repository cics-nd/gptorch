#
# Apr 17, 2017    Yinhao Zhu
#
# class for likelihoods: p(y | f)

from __future__ import absolute_import
from . import densities
from .model import Model, Param

from torch.nn import Parameter
import torch as th

from .settings import DefaultPositiveTransform
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
    2) Do the marginalization in logp(y) = \int logp(y|f) p(f) df
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
        :param var_f: variance of the latent functions
        :return: mean and variance of the observations
        """
        # TODO: Gauss-Hermite quadrature
        raise NotImplementedError

    def forward(self):
        return None

    @abc.abstractmethod
    def propagate_log(self, qf, targets):
        """
        Evaluate the marginal log-likelihood at the targets:
        <log p(y|f)>_q(f)
        Used in variational inference.
        """
        raise NotImplementedError("Implement quadrature fallback")


class Gaussian(Likelihood):
    # Possibly replace these with torch.distributions?
    def __init__(self, variance=1.0):
        super(Gaussian, self).__init__()
        self.variance = Param(th.Tensor([variance]).type(float_type),
                              transform=DefaultPositiveTransform())

    def logp(self, F, Y):
        """
        Evaluate the log-density of targets at Y, given a Gaussian density 
        centered at F.

        :param F: Center of the density
        :type F: torch.autograd.Variable
        :param Y: Targets where we want to compute the log-pdf
        :type Y: torch.autograd.Variable
        """
        return densities.gaussian(F, Y, self.variance.transform())

    def predict_mean_variance(self, mean_f, var_f):
        """
        Integrate the input (a Gaussian with provided mean & variance) over the
        likelihood density

        :param mean_f: Mean of input Gaussian
        :type mean_f: torch.autograd.Variable
        :param var_f: Variance of input Gaussian
        :type var_f: torch.autograd.Variable

        :return: (torch.autograd.Variable, torch.autograd.Variable) mean & var
        """
        # TODO: consider mulit-output case
        # stupid syntax - expecting broadcasting in PyTorch
        return mean_f, var_f + self.variance.transform().expand_as(var_f)

    def predict_mean_covariance(self, mean_f, cov_f):
        return mean_f, cov_f + self.variance.transform().expand_as(cov_f).\
            diag().diag()
