#
# Apr 17, 2017    Yinhao Zhu
#
# class for likelihoods: p(y | f)

from torch.nn import Parameter
import torch
from torch import distributions

from .model import Model, Param
from .settings import DefaultPositiveTransform
from .util import torch_dtype

torch.set_default_dtype(torch_dtype)


class Likelihood(Model):
    """
    Base class for likelihoods, i.e. objects handling the conditional 
    probability relating observed targets y and a (GP) latent function f,
    p(y|f)
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


class Gaussian(Likelihood):
    """
    (Spherical) Gaussian likelihood p(y|f)
    """
    def __init__(self, variance=1.0):
        super(Gaussian, self).__init__()
        self.variance = Param(torch.Tensor([variance]),
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
        return distributions.Normal(F, torch.sqrt(self.variance.transform())). \
            log_prob(Y)

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
