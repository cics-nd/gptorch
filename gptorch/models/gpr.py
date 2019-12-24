#
# Apr 15, 2017  Yinhao Zhu
#
# Class for GP regression

"""
Class for Vanilla GP regression.
"""

import torch
import numpy as np

from .. import kernels
from ..model import Param
from .. import likelihoods
from ..functions import cholesky, inverse, lt_log_determinant, trtrs
from ..util import TensorType
from .base import GPModel


class GPR(GPModel):
    """
    Gaussian Process Regression
    """

    def __init__(self, x, y, kernel, mean_function=None, likelihood=None, name="gpr"):
        """
        Default likelihood is Gaussain, mean function is zero.

        Args:
            observations (np.ndarray or TensorType): a matrix of observed outputs.
                Each datum is a row in the matrix (# rows = # training data,
                # columns = output dimensionality)
            input (np.ndarray or TensorType): the observed inputs.
                If input is a numpy matrix or torch TensorType, then each datum is
                a row in the matrix (# rows = # training data,
                # columns = input dimensionality).
            kernel (Kernel): The kernel function for computing the covariance
                matrices
            mean_function (MeanFunction): class for populating the design matrix
                for computing the parameterized mean function.
            likelihood (Likelihood): A likelihood model
        """
        
        super().__init__(x, y, kernel, likelihood, mean_function, name)

    def log_likelihood(self, x=None, y=None):
        """
        Loss is equal to the negative of the prior log likelihood

        Adapted from Rasmussen & Williams, GPML (2006), p. 19, Algorithm 2.1.
        """

        x = x if x is not None else self.X
        y = y if y is not None else self.Y
        if not x.shape[0] == y.shape[0]:
            raise ValueError("X and Y must have same # data.")

        num_input, dim_output = y.shape

        L = cholesky(self._compute_kyy(x=x))
        alpha = trtrs(y - self.mean_function(x), L)
        const = TensorType([-0.5 * dim_output * num_input * np.log(2 * np.pi)])
        if alpha.is_cuda:
            const = const.cuda()  # TODO cache this?
        loglik = -0.5 * alpha.pow(2).sum() - dim_output * lt_log_determinant(L) + const
        return loglik

    def _compute_kyy(self, x=None):
        """
        Computes the covariance matrix over the training inputs

        Returns:
            Ky (TensorType)
        """

        x = x if x is not None else self.X
        num_input = x.shape[0]

        return (
            self.kernel.K(x)
            + (self.likelihood.variance.transform())
            .expand(num_input, num_input)
            .diag()
            .diag()
        )

    def _predict(self, x_new: TensorType, diag=True, x=None):
        """
        This method computes

        .. math::
            p(F^* | Y )

        where F* are points on the GP at x_new, Y are observations at the
        input X of the training data.
        :param x_new: test inputs; should be two-dimensional
        """

        x = x if x is not None else self.X

        k_ys = self.kernel.K(x, x_new)

        L = cholesky(self._compute_kyy(x=x))
        A = trtrs(k_ys, L)
        V = trtrs(self.Y - self.mean_function(x), L)
        mean_f = A.t() @ V + self.mean_function(x_new)

        if diag:
            var_f = (
                self.kernel.Kdiag(x_new) - 
                (A * A).sum(0)
            )[:, None].expand_as(mean_f)
        else:
            var_f = self.kernel.K(x_new) - A.t() @ A

        return mean_f, var_f
