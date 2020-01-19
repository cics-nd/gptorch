#
# Yinhao Zhu, May 01, 2017
#
"""
Sparse GP regression, including variational GP and others.
"""

from __future__ import absolute_import

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import distributions
from torch.distributions.transforms import LowerCholeskyTransform

from ..model import Param
from ..functions import cholesky, trtrs
from ..mean_functions import Zero
from ..likelihoods import Gaussian
from ..util import TensorType, torch_dtype, as_tensor, kmeans_centers
from ..util import LazyMultivariateNormal
from .gpr import GPR
from .base import GPModel


class _InducingPointsGP(GPModel):
    """
    Parent class for GPs with inducing points
    """

    def __init__(
        self,
        x,
        y,
        kernel,
        num_inducing_points=None,
        inducing_points=None,
        mean_function=None,
        likelihood=None,
    ):
        """
        Assume Gaussian likelihood

        Args:
            observations (np.ndarray): Y, n x p
            input (np.ndarray): X, n x q
            kernel (gptorch.Kernel):
            inducing_points (np.ndarray, optional): Z, m x q
            num_inducing (int), optional): number of inducing inputs

        Input, observations, and kernel must be specified, if both
        ``inducing_points`` and ``num_inducing`` are not set, 1/10 th of total
        points (up to 100) will be draw randomly from input as the inducing 
        points.
        """

        super().__init__(x, y, kernel, likelihood, mean_function)

        if inducing_points is None:
            if num_inducing_points is None:
                num_inducing_points = np.clip(x.shape[0] // 10, 1, 100)
            inducing_points = kmeans_centers(
                x, num_inducing_points, perturb_if_fail=True
            )
            # indices = np.random.permutation(len(x))[:num_inducing_points]
            # inducing_points = TensorType(x[indices])

        # Z stands for inducing input points as standard in the literature
        self.Z = Param(as_tensor(inducing_points))

    @property
    def num_inducing(self) -> int:
        """
        Number of inducing points
        """
        return self.Z.shape[0]


class FITC(_InducingPointsGP):
    """
    Fully Independent Training Conditional approximation for GP

    References:
        Snelson, Edward, and Zoubin Ghahramani. "Sparse Gaussian processes
         using pseudo-inputs." Advances in neural information processing
         systems 18 (2006): 1257.
        Quinonero-Candela, Joaquin, and Carl Edward Rasmussen. "A unifying
        view of sparse approximate Gaussian process regression." Journal of
        Machine Learning Research 6.Dec (2005): 1939-1959.
    """

    # TODO: add FITC for sparse GP regression
    pass


class VFE(_InducingPointsGP):
    """
    Variational Free Energy approximation for GP

    Reference:
        Titsias, Michalis K. "Variational Learning of Inducing Variables
        in Sparse Gaussian Processes." AISTATS. Vol. 5. 2009.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self.mean_function, Zero
        ), "Mean functions not implemented for VFE yet."

    def log_likelihood(self, x=None, y=None):
        """
        Computes the variational lower bound of the true log marginal likelihood
        Eqn (9) in Titsias, Michalis K. "Variational Learning of Inducing Variables
        in Sparse Gaussian Processes." AISTATS. Vol. 5. 2009.
        """

        x = x if x is not None else self.X
        y = y if y is not None else self.Y
        if not x.shape[0] == y.shape[0]:
            raise ValueError("X and Y must have same # data.")

        num_inducing = self.num_inducing
        num_data = x.shape[0]
        d_out = self.output_dimension
        # TODO: add mean_functions
        # err = self.Y - self.mean_function(x)
        err = self.Y
        Kff_diag = self.kernel.Kdiag(x)
        Kuf = self.kernel.K(self.Z, x)
        # add jitter
        Kuu = self.kernel.K(self.Z)
        L = cholesky(Kuu)

        A = trtrs(Kuf, L)
        AAT = A @ A.t() / self.likelihood.variance.transform().expand_as(Kuu)
        B = AAT + torch.eye(num_inducing, dtype=torch_dtype).to(AAT.device)
        LB = cholesky(B)
        # divide variance at the end
        c = trtrs(A @ err, LB) / self.likelihood.variance.transform()

        # Evidence lower bound
        elbo = TensorType([-0.5 * d_out * num_data * np.log(2 * np.pi)]).to(c.device)
        elbo -= d_out * LB.diag().log().sum()
        elbo -= 0.5 * d_out * num_data * self.likelihood.variance.transform().log()
        elbo -= (
            0.5
            * (err.pow(2).sum() + d_out * Kff_diag.sum())
            / self.likelihood.variance.transform()
        )
        elbo += 0.5 * c.pow(2).sum()
        elbo += 0.5 * d_out * AAT.diag().sum()

        return elbo[0]

    def _predict(self, x_new: TensorType, diag=True, x=None):
        """
        Compute posterior p(f*|y), integrating out induced outputs' posterior.

        :return: (mean, var/cov)
        """

        x = x if x is not None else self.X

        z = self.Z
        z.requires_grad_(False)

        num_inducing = z.size(0)

        # err = self.Y - self.mean_function(x)
        err = self.Y
        Kuf = self.kernel.K(z, x)
        # add jitter
        Kuu = self.kernel.K(z)
        Kus = self.kernel.K(z, x_new)
        L = cholesky(Kuu)
        A = trtrs(Kuf, L)
        AAT = A @ A.t() / self.likelihood.variance.transform().expand_as(Kuu)
        B = AAT + torch.eye(num_inducing, dtype=torch_dtype).to(AAT.device)
        LB = cholesky(B)
        # divide variance at the end
        c = trtrs(A @ err, LB) / self.likelihood.variance.transform()
        tmp1 = trtrs(Kus, L)
        tmp2 = trtrs(tmp1, LB)
        mean = tmp2.t() @ c

        if diag:
            var = (
                self.kernel.Kdiag(x_new)
                - tmp1.pow(2).sum(0).squeeze()
                + tmp2.pow(2).sum(0).squeeze()
            )[:, None].expand_as(mean)
            return distributions.Normal(mean, var.sqrt())
        else:
            cov = self.kernel.K(x_new) + tmp2.t() @ tmp2 - tmp1.t() @ tmp1
            return LazyMultivariateNormal(mean.T, cov)


def minibatch(loss_func):
    """
    Decorator to use minibatching for a loss function (e.g. SVGP)
    """

    def wrapped(obj, x=None, y=None):
        if x is not None:
            assert y is not None
        else:
            # Get from model:
            if obj.batch_size is not None:
                i = np.random.permutation(obj.num_data)[: obj.batch_size]
                x, y = obj.X[i, :], obj.Y[i, :]
            else:
                x, y = obj.X, obj.Y

        return loss_func(obj, x, y)

    return wrapped


class SVGP(_InducingPointsGP):
    """
    Sparse variational Gaussian process.

    James Hensman, Nicolo Fusi, and Neil D. Lawrence,
    "Gaussian processes for Big Data" (2013)

    James Hensman, Alexander Matthews, and Zoubin Ghahramani, 
    "Scalable variational Gaussian process classification", JMLR (2015).
    """

    def __init__(
        self,
        x,
        y,
        kernel,
        num_inducing_points=None,
        inducing_points=None,
        mean_function=None,
        likelihood=Gaussian(),
        batch_size=None,
    ):
        """
        :param batch_size: How many points to process in a minibatch of 
            training.  If None, no minibatches are used.
        """
        super().__init__(
            x,
            y,
            kernel,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            mean_function=mean_function,
            likelihood=likelihood,
        )
        # assert batch_size is None, "Minibatching not supported yet."
        self.batch_size = batch_size

        # Parameters for the (Gaussian) variational posterior over the induced
        # outputs.
        # Note: induced_output_mean does NOT include the contribution due to the
        # mean function.
        self.induced_output_mean = Param(
            torch.zeros(self.num_inducing, self.output_dimension, dtype=torch_dtype)
        )
        self.induced_output_chol_cov = Param(
            cholesky(self.kernel.K(self.Z)), transform=LowerCholeskyTransform()
        )

    @minibatch
    def log_likelihood(self, x, y):
        """
        Variational bound.
        """

        if not x.shape[0] == y.shape[0]:
            raise ValueError("X and Y must have same # data.")

        chol_kuu = cholesky(self.kernel.K(self.Z))

        # Marginal posterior q(f)'s mean & variance
        pf = self._predict(x, diag=True, chol_kuu=chol_kuu)
        marginal_log_likelihood = self.likelihood.marginal_log_likelihood(pf, y)
        # Account for size of minibatch relative to the total dataset size:
        marginal_log_likelihood *= self.num_data / x.shape[0]

        mu_xu = self.mean_function(self.Z)  # Prior mean
        qu_mean = self.induced_output_mean + mu_xu
        qu_lc = self.induced_output_chol_cov.transform()
        # Each output dimension has its own Multivariate normal (different
        # means, shared covariance); the joint distribution is the product
        # across output dimensions.
        #
        # Transpose means since MultivariateNormal assumes locs of shape [D x N]
        # scale_tril is automatically broadcasted to [D x N x N]
        kl = distributions.kl_divergence(
            distributions.MultivariateNormal(qu_mean.T, scale_tril=qu_lc),
            distributions.MultivariateNormal(mu_xu.T, scale_tril=chol_kuu),
        )[0]

        return marginal_log_likelihood - kl

    def _predict(self, x_new: TensorType, diag=True, chol_kuu=None, **kwargs):
        """
        SVGP Prediction uses inducing points as sufficient statistics for the 
        posterior.

        Could implement Marginalization of Gaussians (cf. PRML p. 93), but
        something specific to (positive-definite) kernel matrices should 
        perform better.

        Shapes of outputs are:
        diag: both are [N x dy]
        not diag: mean is [N x dy], cov is [N x N]

        :param x_new: inputs to predict on.
        :param diag: if True, return variance of prediction; False=full cov
        :param chol_kuu: The Cholesky of the kernel matrix for the inducing 
            inputs (to enable reuse when computing the training loss)
    
        :return: (torch.Tensor, torch.Tensor) mean & [co]variance
        """

        chol_kuu = cholesky(self.kernel.K(self.Z)) if chol_kuu is None else chol_kuu
        kuf = self.kernel.K(self.Z, x_new)
        alpha = trtrs(kuf, chol_kuu).t()
        # beta @ beta.t() = inv(L) @ S @ inv(L'), S=post cov of induced outs
        beta = trtrs(self.induced_output_chol_cov.transform(), chol_kuu)
        mu_x = self.mean_function(x_new)

        # Remember: induced_output_mean doesn't include mean function, so no
        # need to subtract it.
        f_mean = alpha @ trtrs(self.induced_output_mean, chol_kuu) + mu_x

        # gamma @ gamma.t() = Kfu @ inv(Kuu) @ S @ inv(Kuu) @ Kuf
        gamma = alpha @ beta

        if diag:
            f_var = (
                self.kernel.Kdiag(x_new)
                - torch.sum(alpha ** 2, dim=1)
                + torch.sum(gamma ** 2, dim=1)
            )[:, None].expand_as(f_mean)
            return distributions.Normal(f_mean, f_var.sqrt())
        else:
            f_cov = self.kernel.K(x_new) - alpha @ alpha.t() + gamma @ gamma.t()
            return LazyMultivariateNormal(f_mean.T, f_cov)
