#
# Apr 15, 2017  Yinhao Zhu
#
# Class for GP regression

"""
Class for Vanilla GP regression.
"""

from __future__ import absolute_import
from gptorch import kernels
from gptorch.model import GPModel, Param
import gptorch.likelihoods
from gptorch.functions import trtrs, cholesky, inverse, lt_log_determinant

import warnings
import torch as th
import numpy as np
from torch.autograd import Variable

tensor_type = th.DoubleTensor


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
            likelihood = gptorch.likelihoods.Gaussian()
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
        alpha = trtrs(L, self.Y)
        const = Variable(th.Tensor([-0.5 * dim_output * num_input * \
                                    np.log(2 * np.pi)]).type(tensor_type))
        loss = 0.5 * alpha.pow(2).sum() + dim_output * lt_log_determinant(L) \
            - const
        return loss
        return None if self.mean_function is None \
            else self.mean_function(self.X)

    def _compute_kyy(self):
        """
        Computes the covariance matrix over the training inputs

        Returns:
            Ky (th.Tensor)
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
            # output is a data matrix, rows correspond to the rows in input,
            # columns are treated independently
            input_new = Variable(th.Tensor(input_new).type(tensor_type),
                                 requires_grad=False, volatile=True)

        k_ys = self.kernel.K(self.X, input_new)
        kyy = self._compute_kyy()

        L = cholesky(kyy)
        A = trtrs(L, k_ys)
        V = trtrs(L, self.Y)
        mean_f = th.mm(th.transpose(A, 0, 1), V)

        if self.mean_function is not None:
            mean_f += self.mean_function(input_new)

        var_f_1 = self.kernel.Kdiag(input_new) if diag else \
            self.kernel.K(input_new)  # Kss

        if diag:
            var_f_2 = th.sum(A * A, 0)
        else:
            var_f_2 = th.mm(A.t(), A)
        var_f = var_f_1 - var_f_2

        return mean_f, var_f
