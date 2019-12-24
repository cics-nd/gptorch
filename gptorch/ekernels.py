"""
Kernel expectations

Supports RBF and Linear kernels

on Xcov: posterior covariance matrices for X

In Sheffield GPLVM, the multi-output GP has independent output.
Thus for each datum X[i, :], it has a diagonal covariance matrix whether
X is iid or sequential data.

But there could be the case in the future that each datum x[i, :] has a
full covariance matrix. This will be left as a future feature.
currently only support the diagonal covariance for each datum, though
preprocessing of Xcov is needed for sequential data, as follows:

Lamda: n x q
Xcov: q x n x n
    eqn (3.30) (similar thing for Xmean)
    Xcov[i, :, :] = (Kx.inverse + Lamda[:, i].diag).inverse
    
For the computation in ekernels, only the diagonal part of each 
Xcov[i, :, :] (n x n) is needed.
Xcov[i, :] <- Xcov[i].diag()
Xcov <- Xcov.t()

Not sure if those slice operations are supported by autograd, to check!

GPflow's implementation takes too much memory, but parallel.
Our current implementation is serial, small memory, could be distributed.
For n = 1000, m = 50, q = 10. GPtorch takes 2s to forward, and 1.5s backward.
Comparable to GPflow at this scale.
"""

# Xcov.transform() --> Xcov

# July 20
# add more parallel implementation (more memory consumption, but faster)
# the covariance matrix can be full matrix, not diagonal


from __future__ import absolute_import
import gptorch.kernels
import torch as th
from torch.autograd import Variable
from gptorch.functions import cholesky, trtrs

float_type = th.DoubleTensor

# TODO: necesary to be able to handle any type of kernels (BB)


class Rbf(gptorch.kernels.Rbf):
    def eKxx(self, X):
        """psi_0 in GPy.

        Args:
            X (np.ndarray or Variable): input points of GP

        Returns:
            (Variable): an array with elements are all the kernel variance
        """

        return self.Kdiag(X)

    def eKxz(self, Z, Xmean, Xcov, requires_transform=True):
        """Computes the expectation, eKzxKxz_1, n x m:

        .. math::
            <K_{xz}>_{q(X)}

        Closed form solution for RBF kernel expectation is in the appendix B.2
        of Damianou's Diss.

        Args:
            Z (Variable): m x q inducing input
            Xmean (Variable): n x q mean of input X
            Xcov (Varible): n x q posterior covariance of X, for both iid and
            seq
            
        """
        num_data = Xmean.size(0)
        dim_latent = Xmean.size(1)
        num_inducing = Z.size(0)
        length_scales = self.length_scales.transform()
        variance = self.variance.transform()

        if requires_transform:
            Xcov = Xcov.transform()

        # compute scaled squared distance matrix
        Xmean = Xmean / th.sqrt(length_scales.pow(2).expand_as(Xmean) + Xcov)
        # every row of Xmean_2 is the repetition of the Euclidean norm of
        # Xmean[i, :], n x m
        Xmean_2 = Xmean.pow(2).sum(1).view(-1, 1).expand(num_data, num_inducing)
        # compute the eKxz[i, :] by iterating over each datum,
        # could be distributed
        eKxz_list = []
        for i in xrange(num_data):
            Zi = Z / th.sqrt(
                length_scales.pow(2).expand_as(Z) + Xcov[i, :].expand_as(Z)
            )
            Xmean_Zi = Xmean[i, :].unsqueeze(0).mm(Zi.t())
            # Zi_2 = Zi.pow(2).sum(1).t()
            Zi_2 = Zi.pow(2).sum(1).view(1, -1)
            log_term = (
                (
                    Xcov[i, :] / length_scales.pow(2)
                    + Variable(th.ones(dim_latent).type(float_type))
                )
                .log()
                .sum()
                .expand_as(Xmean_Zi)
            )
            eKxz_list.append(
                Xmean_2[i, :].unsqueeze(0) + Zi_2 - 2 * Xmean_Zi + log_term
            )

        return th.exp(-0.5 * th.cat(eKxz_list, 0)) * variance.expand(
            num_data, num_inducing
        )

    def eKxz_parallel(self, Z, Xmean, Xcov):
        # TODO: add test
        """Parallel implementation (needs more space, but less time)
        Refer to GPflow implementation

        Args:
            Args:
            Z (Variable): m x q inducing input
            Xmean (Variable): n x q mean of input X
            Xcov (Varible): posterior covariance of X
                two sizes are accepted:
                    n x q x q: each q(x_i) has full covariance
                    n x q: each q(x_i) has diagonal covariance (uncorrelated),
                        stored in each row
        Returns:
            (Variable): n x m
        """

        # Revisit later, check for backward support for n-D tensor
        n = Xmean.size(0)
        q = Xmean.size(1)
        m = Z.size(0)
        if Xcov.dim() == 2:
            # from flattered diagonal to full matrix
            cov = Variable(th.Tensor(n, q, q).type(float_type))
            for i in range(Xmean.size(0)):
                cov[i] = Xcov[i].diag()
            Xcov = cov
            del cov
        length_scales = self.length_scales.transform()
        Lambda = length_scales.pow(2).diag().unsqueeze(0).expand_as(Xcov)
        L = cholesky(Lambda + Xcov)
        xz = Xmean.unsqueeze(2).expand(n, q, m) - Z.unsqueeze(0).expand(n, q, m)
        Lxz = th.triangular_solve(xz, L, upper=False)[0]
        half_log_dets = L.diag().log().sum(1) - length_scales.log().sum().expand(n)

        return self.variance.transform().expand(n, m) * th.exp(
            -0.5 * Lxz.pow(2).sum(1) - half_log_dets.expand(n, m)
        )

    def eKzxKxz(self, Z, Xmean, Xcov, requires_transform=True, sum=True):
        """Computes the expectation, psi_2, m x m:

        .. math::
            <K_{zx}L_{xz}>_{q(X)}

        Args:
            Z (Variable): m x q inducing input
            Xmean (Variable): n x q mean of input X
            Xcov (Varible): n x q posterior covariance of X, for both iid and seq
            requires_transform (bool, optional): True if ``Xcov`` needs transform
            sum (bool, optional): False if returning a list of (Psi_2)_i,
                True if returning the sum of (Psi_2)_i, i=1,..,n
        """
        num_data = Xmean.size(0)
        dim_latent = Xmean.size(1)
        length_scales = self.length_scales.transform()
        variance = self.variance.transform()
        if requires_transform:
            Xcov = Xcov.transform()

        # compute shared term, for all i
        eKzxKxz = Variable(th.zeros(Z.size(0), Z.size(0)).type(float_type))
        if not sum:
            eKzxKxz_list = []
        # Z1 is only scaled by length_scales, as the first term of the exponent
        Z1 = Z / length_scales.expand_as(Z)
        ZZT = Z1.mm(Z1.t())
        Z1_2 = Z1.pow(2).sum(1).expand_as(ZZT)
        shared_term = Z1_2 + Z1_2.t() - 2 * ZZT
        # \Xmean_{i,j}
        Xmean = Xmean / th.sqrt(length_scales.pow(2).expand_as(Xmean) + 2 * Xcov)
        Xmean_2 = Xmean.pow(2).sum(1).view(-1, 1)

        for i in xrange(num_data):
            Zi = Z / th.sqrt(
                length_scales.pow(2).expand_as(Z) + 2 * Xcov[i, :].expand_as(Z)
            )
            Zi_2 = Zi.pow(2).sum(1).expand_as(eKzxKxz)

            Xmean_Zi = Zi.mm(Xmean[i, :].unsqueeze(1)).expand_as(eKzxKxz) + Xmean[
                i, :
            ].unsqueeze(0).mm(Zi.t()).expand_as(eKzxKxz)

            log_term = (
                th.log(
                    Xcov[i, :] * 2 / length_scales.pow(2)
                    + Variable(th.ones(dim_latent).type(float_type))
                )
                .sum()
                .expand_as(eKzxKxz)
            )

            eKzxKxz_i = th.exp(
                -0.25 * shared_term
                - Xmean_2[i, :].expand_as(eKzxKxz)
                - 0.25 * (Zi_2 + Zi_2.t() + 2 * Zi.mm(Zi.t()))
                + Xmean_Zi
                - 0.5 * log_term
            ) * variance.pow(2).expand_as(eKzxKxz)
            if sum:
                eKzxKxz += eKzxKxz_i
            else:
                eKzxKxz_list.append(eKzxKxz_i)

        return eKzxKxz if sum else eKzxKxz_list


class Linear(gptorch.kernels.Rbf):
    pass
