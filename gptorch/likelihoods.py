"""
Likelihood classes (extend `gptorch.model.Model`).

The main three tasks that we do with these (conditional) distributions are:
1. Marginalize over a provided input density, i.e. compute 
   p(y) = int p(y|f) p(f) df.  Accomplished by the .forward() method.
2. Evaluate the logarithm of the marginal probability at some provided targets.
   This is used in exact inference (the logarithm is outside of the 
   expectation).  Accomplished by the .log_marginal_likelihood() method.
3. Evaluate the marginal log-likelihood at some provided targets.  This is used 
   in variational inference (the logarithm is inside the expectation). 
   Accomplished by the .marginal_log_likelihood() method.
"""

import abc
from math import pi

from torch.nn import Parameter
import torch
from torch import distributions

from .functions import cholesky
from .model import Model, Param
from .settings import DefaultPositiveTransform
from .util import torch_dtype, TensorType, LazyMultivariateNormal


class Likelihood(Model):
    """
    Probabilities that conventionally factorize across data.
    Typically used as the "second stage" of a GP model that goes 
    x -(GP)-> f -(likelihood)-> y
    x = inputs
    f = latent outputs
    y = observed outputs

    We typically have two uses for these:
    1) Do the marginalization in p(y|x) = int p(y|f) p(f|x) df 
        (e.g. predictions)
    2) Do the marginalization in logp(y) >= int logp(y|f) p(f) df
        (e.g. variational inference)
    """

    def __init__(self):
        super(Likelihood, self).__init__()

        # TODO this could be implemented a bit better to prevent mistakes...
        self._forward_methods = {}
        self._marginal_log_likelihood_methods = {
            torch.Tensor: self.log_marginal_likelihood
        }

    def forward(self, pf: distributions.Distribution):
        """
        Marginalize our likelihood's conditional distribution over the inputted 
        distribution pf:

        p(y) = int p(y|f) p(f) df
        """
        return self._forward_methods[type(pf)](pf)

    def marginal_log_likelihood(
        self, pf: distributions.Distribution, targets: TensorType
    ) -> TensorType:
        """
        Compute the marginal log-likelihood at the provided targets:

        <log p(y|f) >_p(f)

        This is used during variational inference and should not be confused 
        with the log marginal likelihood (used in exact inference)!

        For most combinations of p(f) and p(y|f), this doesn't have an 
        analytical solution.  However, due to the fact that the logarithm is 
        inside the expectation, it can often be efficiently approximated with 
        one-dimensional quadrature, which is implemented by this parent class.

        :param pf: The distribution over latent outputs f to be marginalized 
        over
        :param targets: Locations in output space at which the log-likelihoods
        will be evaluated.  Matrix of shape [N x DY].

        :return: MLL (a scalar)
        """

        return (
            self._marginal_log_likelihood_methods[type(pf)](pf, targets)
            if type(pf) in self._marginal_log_likelihood_methods
            else self._marginal_log_likelihood_monte_carlo(pf, targets)
        )

    @abc.abstractmethod
    def log_marginal_likelihood(
        self, pf: distributions.Distribution, targets: TensorType
    ) -> TensorType:
        """
        Compute the logarithm of the marginal likelihood at the provided 
        targets:

        log[<p(y|f)>_p(f)]

        This is used during exact inference and should not be confused 
        with the marginal log-likelihood (used in variational inference)!

        :param pf: The distribution over latent outputs f to be marginalized 
        over
        :param targets: Locations in output space at which the log-likelihoods
        will be evaluated.  Matrix of shape [N x DY].

        :return: MLL (a scalar)
        """

        raise NotImplementedError()

    def _marginal_log_likelihood_monte_carlo(self, pf, targets, n=100):
        """
        Monte Carlo fallback
        """

        return torch.stack(
            [self(pf.sample()).log_prob(targets).sum() for _ in range(n)]
        ).mean()


class Gaussian(Likelihood):
    """
    (Spherical) Gaussian likelihood p(y|f)
    """

    def __init__(self, variance=1.0):
        super(Gaussian, self).__init__()
        self.variance = Param(
            TensorType([variance]), transform=DefaultPositiveTransform()
        )

        # Would like to replace this with e.g. @register_forward_method on the
        # methods instead.
        self._forward_methods.update(
            {
                torch.Tensor: self._forward_tensor,
                distributions.Normal: self._forward_normal,
                LazyMultivariateNormal: self._forward_multivariate_normal,
            }
        )
        self._marginal_log_likelihood_methods.update(
            {
                distributions.Normal: self._marginal_log_likelihood_gaussian,
                distributions.MultivariateNormal: self._marginal_log_likelihood_gaussian,
            }
        )

    def log_marginal_likelihood(
        self, pf: distributions.Distribution, targets: TensorType
    ) -> TensorType:
        py = self(pf)
        if isinstance(py, distributions.MultivariateNormal):
            # Have to transopse targets because loc parameter for torch MVN is
            # [D x N] when multi-output, no [N x D] as is gptorch's convention.
            return py.log_prob(targets.T).sum()
        else:
            return py.log_prob(targets).sum()

    def _forward_tensor(self, f: TensorType) -> distributions.Normal:
        """
        Propagating just a point
        """
        return distributions.Normal(f, self.variance.transform().sqrt())

    def _forward_normal(self, pf: distributions.Normal) -> distributions.Normal:
        """
        Propagate independent normals through the likelihood
        """
        return distributions.Normal(
            pf.loc, (pf.variance + self.variance.transform()).sqrt()
        )

    def _forward_multivariate_normal(
        self, pf: LazyMultivariateNormal
    ) -> distributions.MultivariateNormal:
        variance = self.variance.transform()
        device = variance.device
        # Do Cholesky using our safe op instead of trusting PyTorch's default.
        scale_tril = cholesky(
            pf.covariance_matrix
            + variance
            * torch.eye(
                *pf.covariance_matrix.shape[-2:], dtype=torch_dtype, device=device
            )
        )
        return distributions.MultivariateNormal(pf.loc, scale_tril=scale_tril)

    def _marginal_log_likelihood_gaussian(self, pf, targets):
        if isinstance(pf, TensorType):
            return self.log_marginal_likelihood(pf, targets)
        if not isinstance(pf, torch.distributions.Normal) and not isinstance(
            pf, torch.distributions.MultivariateNormal
        ):
            # Quadrature fallback
            return super().marginal_log_likelihood(pf, targets)

        mu, s = pf.loc, pf.variance
        n = targets.nelement()
        if not mu.nelement() == n:
            raise ValueError(
                "Targets (%i) and p(f) (%i) have mismatch in size" % (n, mu.nelement())
            )

        sigma_y = self.variance.transform()

        return (
            -0.5
            * (
                n
                * (
                    torch.log(TensorType([2.0 * pi])).to(sigma_y.device)
                    + torch.log(sigma_y)
                )
                + (torch.sum((targets - mu) ** 2) + s.sum()) / sigma_y
            ).sum()
        )


class Bernoulli(Likelihood):
    """
    Bernoulli likelihood:

    p(y=1) = theta
    p(y=0) = 1 - theta

    where (theta in [0,1]) is related to the inputted variable f through some 
    link function that maps from the reals to the unit interval.
    """

    def __init__(self, link: torch.nn.Module = None):
        """
        :param link: the link function relating the latent output to the 
        parameter of the Bernoulli
        """

        super().__init__()
        self.link = torch.nn.Sigmoid() if link is None else link
        self._forward_methods.update({TensorType: self._forward_tensor})

    def _forward_tensor(self, f):
        return distributions.Bernoulli(probs=self.link(f))


class Binomial(Likelihood):
    """
    Binomial likelihood with parameter n:
    $y in [0, dots, n]$

    p(y) = Bin(y; n, p)

    where (theta in [0,1]) is related to the inputted variable f through some 
    link function that maps from the reals to the unit interval.
    """

    def __init__(self, n: int, link: torch.nn.Module = None):
        """
        :param link: the link function relating the latent output to the 
        parameter of the Bernoulli
        """
        super().__init__()

        self.n = n
        self.link = torch.nn.Sigmoid() if link is None else link
        self._forward_methods.update({TensorType: self._forward_tensor})

    def _forward_tensor(self, f):
        return distributions.Binomial(self.n, probs=self.link(f))
