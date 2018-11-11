#
# Yinhao Zhu, May 01, 2017
#
"""
Sparse GP regression, including variational GP and others.
"""

from __future__ import absolute_import
from gptorch.model import GPModel, Param
from gptorch.mean_functions import Zero
from gptorch.likelihoods import Gaussian
from gptorch.functions import cholesky, trtrs

import torch as th
import numpy as np
from torch.autograd import Variable

float_type = th.DoubleTensor


class FITC(GPModel):
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


class VFE(GPModel):
    """
    Variational Free Energy approximation for GP

    Reference:
        Titsias, Michalis K. "Variational Learning of Inducing Variables
        in Sparse Gaussian Processes." AISTATS. Vol. 5. 2009.
    """
    def __init__(self, observations, input, kernel, inducing_points=None,
                 num_inducing=None, mean_function=None, name='variational_gp'):
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
         points will be draw randomly from input as the inducing points.
        """
        likelihood = Gaussian()
        super(VFE, self).__init__(observations, input, kernel, likelihood,
                                  mean_function, name)
        if inducing_points is not None:
            if isinstance(inducing_points, np.ndarray):
                # inducing points are free variational parameters, no constraints
                # However, it is possible those points are optimized to outer regions
                inducing_points = Param(th.Tensor(inducing_points).type(float_type))
        else:
            if num_inducing is None:
                num_inducing = np.max([len(input) // 10, 1])
            # randomly select num_inducing points from input
            indices = np.arange(len(input))
            np.random.shuffle(indices)
            inducing_points = Param(th.Tensor(input[indices[:num_inducing]]).\
                type(float_type))

        self.jitter = Param(th.Tensor([1e-4]).type(float_type), 
                            requires_transform=True)
        # Z stands for inducing points as standard in the literature
        self.Z = inducing_points

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
        Kuu = self.kernel.K(self.Z) + \
              self.jitter.transform().expand(num_inducing).diag()
        L = cholesky(Kuu)

        A = trtrs(L, Kuf)
        AAT = A.mm(A.t()) / self.likelihood.variance.transform().expand_as(Kuu)
        B = AAT + Variable(th.eye(num_inducing).type(float_type))
        LB = cholesky(B)
        # divide variance at the end
        c = trtrs(LB, A.mm(err)) \
            / self.likelihood.variance.transform().expand(num_inducing, dim_output)

        # Evidence lower bound
        elbo = Variable(th.Tensor([-0.5 * dim_output * num_training
                                   * np.log(2 * np.pi)]).type(float_type))
        elbo -= dim_output * LB.diag().log().sum()
        elbo -= 0.5 * dim_output * num_training * self.likelihood.variance.transform().log()
        elbo -= 0.5 * (err.pow(2).sum() + dim_output * Kff_diag.sum()) \
                / self.likelihood.variance.transform()
        elbo += 0.5 * c.pow(2).sum()
        elbo += 0.5 * dim_output * AAT.diag().sum()

        return - elbo

    def _predict(self, input_new, diag=True):
        # following GPflow implementation
        # integrating the inducing variables out

        if isinstance(input_new, np.ndarray):
            # set input_new to be volatile for inference mode
            input_new = Variable(th.Tensor(input_new).type(float_type), volatile=True)

        self.X.volatile = True
        self.Y.volatile = True
        self.Z.volatile = True

        num_inducing = self.Z.size(0)
        dim_output = self.Y.size(1)

        # err = self.Y - self.mean_function(self.X)
        err = self.Y
        # Kff_diag = self.kernel.Kdiag(self.X)
        Kuf = self.kernel.K(self.Z, self.X)
        # add jitter
        # Kuu = self.kernel.K(self.Z) + Variable(th.eye(num_inducing).float() * 1e-5)
        Kuu = self.kernel.K(self.Z) + self.jitter.transform().expand(num_inducing).diag()
        Kus = self.kernel.K(self.Z, input_new)
        L = cholesky(Kuu)
        A = trtrs(L, Kuf)
        AAT = A.mm(A.t()) / self.likelihood.variance.transform().expand_as(Kuu)
        B = AAT + Variable(th.eye(num_inducing).type(float_type))
        LB = cholesky(B)
        # divide variance at the end
        c = trtrs(LB, A.mm(err)) \
            / self.likelihood.variance.transform().expand(num_inducing, dim_output)
        tmp1 = trtrs(L, Kus)
        tmp2 = trtrs(LB, tmp1)
        mean = tmp2.t().mm(c)

        if diag:
            var = self.kernel.Kdiag(input_new) - tmp1.pow(2).sum(0).squeeze() \
                  + tmp2.pow(2).sum(0).squeeze()
            # add kronecker product later for multi-output case
        else:
            var = self.kernel.K(input_new) + tmp2.t().mm(tmp2) \
                  - tmp1.t().mm(tmp1)
        # return mean + self.mean_function(input_new), var
        return mean, var
