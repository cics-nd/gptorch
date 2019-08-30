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

from ..model import Param
from ..functions import cholesky, trtrs
from ..mean_functions import Zero
from ..likelihoods import Gaussian
from ..util import TensorType, torch_dtype, as_tensor, kmeans_centers
from ..util import KL_Gaussian
from .base import GPModel


class _InducingPointsGP(GPModel):
    """
    Parent class for GPs with inducing points
    """
    def __init__(self, x, y, kernel, num_inducing_points=None, 
            inducing_points=None, mean_function=None, likelihood=None):
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
            inducing_points = kmeans_centers(x, num_inducing_points, 
                perturb_if_fail=True)
            # indices = np.random.permutation(len(x))[:num_inducing_points]
            # inducing_points = TensorType(x[indices])
            print("Inducing points:\n{}".format(inducing_points))
        
        # Z stands for inducing input points as standard in the literature
        self.Z = Param(as_tensor(inducing_points))
        self.jitter = 1.0e-6
        

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
        assert isinstance(self.mean_function, Zero), \
            "Mean functions not implemented for VFE yet."

    def compute_loss(self):
        """
        Computes the variational lower bound of the true log marginal likelihood
        Eqn (9) in Titsias, Michalis K. "Variational Learning of Inducing Variables
        in Sparse Gaussian Processes." AISTATS. Vol. 5. 2009.
        """

        num_inducing = self.Z.size(0)
        num_training = self.X.size(0)
        dim_output = self.Y.size(1)
        # TODO: add mean_functions
        # err = self.Y - self.mean_function(self.X)
        err = self.Y
        Kff_diag = self.kernel.Kdiag(self.X)
        Kuf = self.kernel.K(self.Z, self.X)
        # add jitter
        Kuu = self.kernel.K(self.Z) + self.jitter * torch.eye(num_inducing, 
            dtype=torch_dtype)
        L = cholesky(Kuu)

        A = trtrs(Kuf, L)
        AAT = A @ A.t() / self.likelihood.variance.transform().expand_as(Kuu)
        B = AAT + torch.eye(num_inducing, dtype=torch_dtype)
        LB = cholesky(B)
        # divide variance at the end
        c = trtrs(A @ err, LB) / self.likelihood.variance.transform()

        # Evidence lower bound
        elbo = TensorType([-0.5 * dim_output * num_training * np.log(2*np.pi)])
        elbo -= dim_output * LB.diag().log().sum()
        elbo -= 0.5 * dim_output * num_training * self.likelihood.variance.transform().log()
        elbo -= 0.5 * (err.pow(2).sum() + dim_output * Kff_diag.sum()) \
                / self.likelihood.variance.transform()
        elbo += 0.5 * c.pow(2).sum()
        elbo += 0.5 * dim_output * AAT.diag().sum()

        return - elbo

    def _predict(self, x_new: TensorType, diag=True):
        """
        Compute posterior p(f*|y), integrating out induced outputs' posterior.

        :return: (mean, var/cov)
        """

        z = self.Z
        z.requires_grad_(False)

        num_inducing = z.size(0)

        # err = self.Y - self.mean_function(self.X)
        err = self.Y
        Kuf = self.kernel.K(z, self.X)
        # add jitter
        Kuu = self.kernel.K(z) + self.jitter * torch.eye(num_inducing, 
            dtype=torch_dtype)
        Kus = self.kernel.K(z, x_new)
        L = cholesky(Kuu)
        A = trtrs(Kuf, L)
        AAT = A @ A.t() / self.likelihood.variance.transform().expand_as(Kuu)
        B = AAT + torch.eye(num_inducing, dtype=torch_dtype)
        LB = cholesky(B)
        # divide variance at the end
        c = trtrs(A @ err, LB) / self.likelihood.variance.transform()
        tmp1 = trtrs(Kus, L)
        tmp2 = trtrs(tmp1, LB)
        mean = tmp2.t() @ c

        if diag:
            var = self.kernel.Kdiag(x_new) - tmp1.pow(2).sum(0).squeeze() \
                  + tmp2.pow(2).sum(0).squeeze()
        else:
            var = self.kernel.K(x_new) + tmp2.t() @ tmp2 - tmp1.t() @ tmp1

        return mean, var


class SVGP(_InducingPointsGP):
    pass
    # """
    # Sparse variational Gaussian process.

    # Sparse GP with 

    # James Hensman, Nicolo Fusi, and Neil D. Lawrence,
    # "Gaussian processes for Big Data" (2013)

    # James Hensman, Alexander Matthews, and Zoubin Ghahramani, 
    # "Scalable variational Gaussian process classification", JMLR (2015).
    # """
    # def __init__(self, x, y, kernel, num_inducing_points=None, 
    #         inducing_points=None, mean_function=None, likelihood=Gaussian(), 
    #         batch_size=None):
    #     """
    #     :param batch_size: How many points to process in a minibatch of 
    #         training.  If None, no minibatches are used.
    #     """
    #     super().__init__(x, y, kernel, num_inducing_points=num_inducing_points, 
    #         inducing_points=inducing_points, mean_function=mean_function, 
    #         likelihood=likelihood)
    #     assert batch_size is None, "Minibatching not supported yet."
    #     self.batch_size = batch_size

    #     # Parameters for the Gaussian variational posterior over the induced
    #     # outputs:
    #     self.induced_output_mean, self.induced_output_chol_cov = \
    #         self._init_posterior()

    # def compute_loss(self, x: TensorType=None, y: TensorType=None) \
    #         -> TensorType:
    #     """
    #     :param x: batch inputs
    #     :param y: batch outputs
    #     """
    #     x, y = self._get_batch(x, y)
    #     qu_mean = self.induced_output_mean
    #     qu_lc = self.induced_output_chol_cov.transform()
    #     m = self.Z.shape[0]

    #     # Get the mean of the marginal q(f)
    #     k_uf = self.kernel.K(self.Z, x)
    #     kuu = self.kernel.K(self.Z) + \
    #         self.jitter * torch.eye(m, dtype=torch_dtype)
    #     kuu_chol = cholesky(kuu)
    #     a = torch.trtrs(k_uf, kuu_chol, upper=False)[0]
    #     f_mean = a.t() @ \
    #         torch.trtrs(qu_mean, kuu_chol, upper=False)[0]

    #     # Variance of the marginal q(f)
    #     b = torch.trtrs(q_cov_chol, kuu_chol, upper=False)[0]
    #     f_var_1 = (a.t() @ a).sum(1)
    #     f_var_2 = (a.t() @ b).sum(1)
    #     f_var = (self.kernel.Kdiag(x) + f_var_1 + f_var_2)[:, None].expand(
    #         *y.shape)

    #     elbo = self.likelihood.propagate(torch.distributions.Normal(f_mean, 
    #         f_var.sqrt()))
        
    #     kl = KL_Gaussian(qu_mean, qu_lc @ qu_lc.t(), 
    #         torch.zeros(*qu_mean.shape, dtype=torch_dtype, kuu)
    #     return elbo - kl
        
    # def _predict(self, input_new):
    #     raise NotImplementedError("")
    
    # def _get_batch(self, x, y):
    #     """
    #     Get the next batch of data for training.
    #     :return: (TensorType, TensorType) inputs, outputs
    #     """
    #     assert not ((x is None) ^ (y is None)), \
    #         "Cannot provide inputs or outputs only in minibatch"
    #     return self.X, self.Y if x is None else x, y

    # def _init_posterior(self):
    #     """
    #     Get an initial guess at the variational posterior over the induced 
    #     outputs.
    #     """
    #     # For the mean, take the nearest points in input space and steal their
    #     # corresponding outputs.  This could be costly if X is very large...
    #     nearest_points = np.argmin(
    #         squared_distance(self.Z, self.X).detach().numpy(), axis=1)
    #     mean = self.X[nearest_points]

    #     # For the covariance, we'll start with 1/100 of the prior kernel
    #     # matrix. (aka 1/10th the Cholesky).
    #     cov = 0.1 * cholesky(self.kernel.K(self.Z))
    #     return mean, cov
