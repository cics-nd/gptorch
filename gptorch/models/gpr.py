#
# Apr 15, 2017  Yinhao Zhu
#
# Class for GP regression

"""
Class for Vanilla GP regression.
"""

from __future__ import absolute_import
import torch
import numpy as np
from torch.autograd import Variable

from .. import kernels
from ..model import GPModel, Param
from .. import likelihoods
from ..functions import cholesky, inverse, lt_log_determinant
from ..util import TensorType


class GPR(GPModel):
    """
    Gaussian Process Regression
    """
    def __init__(self, observations, input, kernel, mean_function=None,
                 likelihood=None, name='gpr'):
        """
        Default likelihood is Gaussain, mean function is zero.

        Args:
            observations (np.ndarray or Variable): a matrix of observed outputs.
                Each datum is a row in the matrix (# rows = # training data,
                # columns = output dimensionality)
            input (np.ndarray or Variable): the observed inputs.
                If input is a numpy matrix or torch Variable, then each datum is
                a row in the matrix (# rows = # training data,
                # columns = input dimensionality).
            kernel (Kernel): The kernel function for computing the covariance
                matrices
            mean_function (MeanFunction): class for populating the design matrix
                for computing the parameterized mean function.
            likelihood (Likelihood): A likelihood model
        """
        if likelihood is None:
            likelihood = likelihoods.Gaussian()
        super().__init__(observations, input, kernel, likelihood,
                                  mean_function, name)

    def compute_loss(self):
        """
        Loss is equal to the negative of the log likelihood

        Adapted from Rasmussen & Williams, GPML (2006), p. 19, Algorithm 2.1.
        """

        num_input = self.Y.size(0)
        dim_output = self.Y.size(1)

        L = cholesky(self._compute_kyy())
        alpha = torch.triangular_solve(self.Y, L, upper=False)[0]
        const = TensorType([-0.5 * dim_output * num_input * np.log(2 * np.pi)])
        loss = 0.5 * alpha.pow(2).sum() + dim_output * lt_log_determinant(L) \
            - const
        return loss

    def _compute_kyy(self):
        """
        Computes the covariance matrix over the training inputs

        Returns:
            Ky (torch.Tensor)
        """
        num_input = self.Y.size(0)

        return self.kernel.K(self.X) + \
        (self.likelihood.variance.transform()).expand(
            num_input, num_input).diag().diag()

    def _predict(self, input_new, diag, full_cov_size_limit=10000):
        """
        This method computes

        .. math::
            p(F^* | Y )

        where F* are points on the GP at input_new, Y are observations at the
        input X of the training data.
        :param input_new: assume to be numpy array, but should be in two dimensional
        """

        if isinstance(input_new, np.ndarray):
            input_new = TensorType(input_new)
            input_new.requires_grad_(False)

        k_ys = self.kernel.K(self.X, input_new)

        L = cholesky(self._compute_kyy())
        A = torch.triangular_solve(k_ys, L, upper=False)[0]
        V = torch.triangular_solve(self.Y, L, upper=False)[0]
        mean_f = A.t() @ V

        if self.mean_function is not None:
            mean_f += self.mean_function(input_new)

        var_f_1 = self.kernel.Kdiag(input_new) if diag else \
            self.kernel.K(input_new)  # Kss

        if diag:
            var_f_2 = (A * A).sum(0)
        else:
            var_f_2 = A.t() @  A
        var_f = var_f_1 - var_f_2

        return mean_f, var_f
