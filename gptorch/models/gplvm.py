"""
Gaussian Process Latent Variable Model (GPLVM)

"""
#
# Yinhao Zhu, May, July 2017
#
#

from warnings import warn

from gptorch.model import GPModel, Param
from gptorch.likelihoods import Gaussian
from gptorch.mean_functions import Zero
from gptorch.functions import cholesky, trtrs
from gptorch import ekernels, kernels
from gptorch import util

import numpy as np
import torch as th
from torch.autograd import Variable
from sklearn.decomposition import PCA
from time import time

try:
    # Python 2
    from future_builtins import filter
except ImportError:
    # Python 3
    pass

float_type = th.DoubleTensor


class GPLVM(GPModel):
    """
    Variational GPLVM

    Reference:
        Damianou, Andreas. Deep Gaussian processes and variational propagation
        of uncertainty. Diss. University of Sheffield, 2015.

    """

    def __init__(
        self,
        observations,
        dim_latent,
        num_inducing,
        Xmean=None,
        inducing_points=None,
        kernel=None,
        kernel_x=None,
        data_type="iid",
        collapsed_bound=True,
        large_p=False,
    ):
        """
        Initialization for the variational GPLVM

        Args:
            observations (np.ndarray): Observed data for unsupervised learning
            dim_latent (int): Dimensionality of the latent variables
            num_inducing (int): Number of inducing points
            Xmean (np.ndarray): Latent variable means (if None, will be init by
                PCA)
            inducing_points (np.ndarray): Inducing points, Z
            kernel (gptorch.Kernel):
            data_type (string): ``iid`` or ``seq`` (sequential)
            collapsed_bound (bool, optional): True for computing the ELBO when
                inducing variables are collapsed (second bound), False for
                the uncollapsed bound
            large_p (bool, optional): True for the case of small n, large p
                (HD video), False for the case of large p, small n.
                This option affects the computation of KL(q(X) || p(X))
        """

        warn("GPLVM is unstable and not recommended for use!")
        
        assert isinstance(
            observations, np.ndarray
        ), "Observation matrix should be a np.ndarray."

        if Xmean is None:
            print("GPLVM: Initialize the Xmean using PCA")
            if large_p:
                pca_sklean = PCA(n_components=dim_latent)
                Xmean = pca_sklean.fit_transform(observations)
            else:
                Xmean = util.as_variable(util.PCA(observations, dim_latent))
        else:
            assert isinstance(Xmean, np.ndarray), (
                "Initialization of posterior mean of latent variables should"
                " be np.ndarray."
            )

        if kernel is None:
            kernel = ekernels.Rbf(dim_latent, ARD=True)
        else:
            assert (
                dim_latent == kernel.input_dim
            ), "Input dimensionality of kernel must be equal to dim_latent."
            assert isinstance(
                kernel, ekernels.Rbf
            ), "Supports only ekernel.Rbf currently."

        super(GPLVM, self).__init__(
            Xmean, observations, kernel, Gaussian(), Zero(), name="GPLVM"
        )
        del self.X

        # flag to distinguish training and testing mode
        self.inference = False
        self.data_type = data_type
        self.is_collapsed = collapsed_bound
        self.is_large_p = large_p

        # Setup for test time inference
        # test data will be assigned in the projection method
        self.Y_test = None
        # latent variable mean and covariance for the test data
        self.Xmean_test = None
        self.Xcov_test = None
        # observed dimensions of the test data
        self.observed_dims = None
        self.saved_terms = {}
        if self.is_large_p:
            # saved for faster computation in the lower bound, n x n
            self.saved_terms["YYT"] = self.Y.mm(self.Y.t())

        if self.data_type == "iid":
            # posterior mean of X initialized by PCA of Y: n x q
            # self.Xmean = Param(th.from_numpy(Xmean).type(float_type))
            self.Xmean = Param(Xmean.data)
            # posterior covariance of X: n x q
            self.Xcov = Param(
                0.5 * th.ones(self.Xmean.size()).type(float_type)
                + 0.001 * th.randn(self.Xmean.size()).type(float_type),
                requires_transform=True,
            )
        else:
            # sequential data
            # temporal kernel for the GP from time t to latent variables X
            if isinstance(kernel_x, kernels.Kernel):
                assert kernel_x.input_dim == 1, (
                    "Currently only supports time input, i.e. kernel with "
                    "one dimension input"
                )
                self.kernel_x = kernel_x
            else:
                # TODO: kernel_x, better initialization needed!
                self.kernel_x = kernels.Rbf(1, variance=0.5, length_scales=0.5)
                self.kernel_x.variance.requires_grad = False

            # 1) vanilla, O(n^2*q) parameters for q(X) (not scalable)
            # ----- 2) Reparameterization (3.30) p58 in Damianou Diss. ------ current impl
            # 3) recognition model (will be the useful one) (RNN)

            # TODO: add the ability to handle multiple sequences (regressive)

            # 2) Reparameterization (3.30) p58 in Damianou Diss.
            # posterior mean of X initialized by PCA of Y: n x q
            # Xmean = th.Tensor(x_post_mean).type(float_type)
            # intermediate variables, useful for inference purpose, or other queries
            self.Xmean = Variable(Xmean)
            # init the cov matrix by using the kernel
            # timestamp is not required for stationary kernels
            Kx = self.kernel_x.K(np.array(xrange(self.Y.size(0)))[:, None])
            # optimization parameters are mu_bar as in (3.30)
            self.Xmean_bar = Param(Kx.data.inverse().mm(Xmean))

            # assume the posterior S is the same as the prior Kx
            # self.lambda_ = Param(th.zeros(Xmean.size()).type(float_type))
            # assume the posterior S is close to the prior Kx
            # Constrain the Lambda to be positive, to ensure the S is PSD
            self.Lambda = Param(
                th.rand(Xmean.size()).type(float_type) * 0.25, requires_transform=True
            )
            # dummy initialization, n x q
            self.Xcov = Variable(th.ones(Xmean.size()).type(float_type) * 0.5)

        if inducing_points is not None:
            if isinstance(inducing_points, np.ndarray):
                assert (
                    inducing_points.shape[0] == num_inducing
                    and inducing_points.shape[1] == dim_latent
                ), "Dimensionality of inducing points does not match"
                self.Z = Param(th.from_numpy(inducing_points).type(float_type))
        else:
            # inducing points Z, init with subset of posterior mean of X
            z_np = Xmean.data.numpy()[
                np.random.choice(Xmean.size(0), num_inducing, replace=False)
            ]
            self.Z = Param(float_type(z_np))

        # Uncollapsed case, the number of parameters associated with inducing points
        #  variance is O(m^2)
        # TODO: stochasitic optimization with the uncollpased bound
        # MNIST data set 60k, 28 x 28 digits
        if not self.is_collapsed:
            # posterior mean of inducing variables U, init with subset of observations
            self.Umean = Param(
                th.from_numpy(
                    self.Y[
                        np.random.choice(self.Y.size(0), num_inducing, replace=False)
                    ]
                ).type(float_type)
            )
            # posterior variance of inducing variables U: m x m
            # needs parameterization of cov matrix, e.g. Chol decomposition
            self.Ucov = Param(
                0.5 * th.ones(num_inducing, num_inducing), requires_transform=True
            )

        # self.jitter = Param(th.FloatTensor([1e-4]), requires_transform=True)
        self.jitter = Variable(th.Tensor([1e-6]).type(float_type))

        # computes the total number of parameters to optimize over
        num_parameters = 0
        for param in self.parameters():
            num_parameters += param.data.numpy().size
        print("GPLVM: Number of optimization parameters is  %d" % num_parameters)

    def log_likelihood(self):
        """
        Computation graph for the ELBO (Evidence Lower Bound) of
        the variational GPLVM
        For the implementation details, please see ``notes/impl_gplvm``.

        """
        num_data = self.Y.size(0)
        dim_output = self.Y.size(1)
        dim_latent = self.Z.size(1)
        num_inducing = self.Z.size(0)

        var_kernel = self.kernel.variance.transform()
        var_noise = self.likelihood.variance.transform()

        # computes kernel expectations
        eKxx = num_data * var_kernel
        if self.data_type == "iid":
            eKxz = self.kernel.eKxz(self.Z, self.Xmean, self.Xcov)
            eKzxKxz = self.kernel.eKzxKxz(self.Z, self.Xmean, self.Xcov)
        else:
            # seq data
            # compute S_j's and mu_bar_j's (reparameterization: forward)
            # self.Xmean, self.Xcov = self._reparam_vargp(self.Xmean_bar, self.Lambda)
            Kx = self.kernel_x.K(np.array(xrange(self.Y.size(0)))[:, None])
            # print(Kx.data.eig())
            Lkx = cholesky(Kx, flag="Lkx")
            # Kx_inverse = inverse(Kx)
            self.Xmean = Kx.mm(self.Xmean_bar)
            Xcov = []
            # S = []
            Le = []
            In = Variable(th.eye(num_data).type(float_type))
            for j in xrange(dim_latent):
                Ej = Lkx.t().mm(self.Lambda.transform()[:, j].diag()).mm(Lkx) + In
                # print(Ej.data.eig())
                Lej = cholesky(Ej, flag="Lej")
                Lsj = trtrs(Lej, Lkx.t()).t()
                Sj = Lsj.mm(Lsj.t())
                Xcov.append(Sj.diag().unsqueeze(1))
                # S.append(Sj)
                Le.append(Lej)
            self.Xcov = th.cat(Xcov, 1)
            eKxz = self.kernel.eKxz(self.Z, self.Xmean, self.Xcov, False)
            eKzxKxz = self.kernel.eKzxKxz(self.Z, self.Xmean, self.Xcov, False)

        # compute ELBO
        # add jitter
        # broadcast update
        Kzz = self.kernel.K(self.Z) + self.jitter.expand(self.Z.size(0)).diag()
        L = cholesky(Kzz, flag="Lkz")
        A = trtrs(L, trtrs(L, eKzxKxz).t()) / var_noise.expand_as(L)
        B = A + Variable(th.eye(num_inducing).type(float_type))
        LB = cholesky(B, flag="LB")

        # log|B|
        # log_det_b = LB.diag().log().sum()

        log_2pi = Variable(th.Tensor([np.log(2 * np.pi)]).type(float_type))
        elbo = -dim_output * (
            LB.diag().log().sum() + 0.5 * num_data * (var_noise.log() + log_2pi)
        )
        elbo -= 0.5 * dim_output * (eKxx / var_noise - A.trace())

        if not self.is_large_p:
            # distributed
            # C = Variable(th.zeros(num_inducing, dim_output))
            # for i in xrange(num_data):
            #     C += Psi[i, :].unsqueeze(1).mm(self.Y[i, :].unsqueeze(0))
            C = eKxz.t().mm(self.Y)
            D = trtrs(LB, trtrs(L, C))
            elbo -= (
                0.5
                * (
                    self.Y.t().mm(self.Y) / var_noise.expand(dim_output, dim_output)
                    - D.t().mm(D) / var_noise.pow(2).expand(dim_output, dim_output)
                ).trace()
            )
        else:
            # small n, pre-compute YY'
            # YYT = self.Y.mm(self.Y.t())
            D = trtrs(LB, trtrs(L, eKxz.t()))
            W = Variable(th.eye(num_data).type(float_type)) / var_noise.expand(
                num_data, num_data
            ) - D.t().mm(D) / var_noise.pow(2).expand(num_data, num_data)
            elbo -= 0.5 * (W.mm(self.saved_terms["YYT"])).trace()

        # KL Divergence (KLD) btw the posterior and the prior
        if self.data_type == "iid":
            const_nq = Variable(th.Tensor([num_data * dim_latent]).type(float_type))
            # eqn (3.28) below p57 Damianou's Diss.
            KLD = 0.5 * (
                self.Xmean.pow(2).sum()
                + self.Xcov.transform().sum()
                - self.Xcov.transform().log().sum()
                - const_nq
            )
        else:
            # seq data (3.29) p58
            # Xmean n x q
            # S: q x n x n
            # Kx, Kx_inverse
            KLD = Variable(th.Tensor([-0.5 * num_data * dim_latent]).type(float_type))
            KLD += 0.5 * self.Xmean_bar.mm(self.Xmean_bar.t()).mm(Kx.t()).trace()
            for j in xrange(dim_latent):
                Lej_inv = trtrs(Le[j], In)
                KLD += 0.5 * Lej_inv.t().mm(Lej_inv).trace() + Le[j].diag().log().sum()

        elbo -= KLD
        return elbo

    def log_likelihood_inference(self):
        """Computes the loss in the inference mode, e.g. for projection.
        Handles both fully observed and partially observed data.

        Only iid latent is implemented.
        """
        num_data_train = self.Y.size(0)
        # dim_output_train = self.Y.size(1)
        dim_latent = self.Z.size(1)
        num_inducing = self.Z.size(0)
        num_data_test = self.Y_test.size(0)
        # total number of data for inference
        num_data = num_data_train + num_data_test
        # dimension of output in the test time
        dim_output = self.Y_test.size(1)
        # whole data for inference
        if self.observed_dims is None:
            Y = th.cat((self.Y, self.Y_test), 0)
        else:
            Y = th.cat((self.Y.index_select(1, self.observed_dims), self.Y_test), 0)

        var_kernel = self.kernel.variance.transform()
        var_noise = self.likelihood.variance.transform()

        # computes kernel expectations
        # eKxx = num_data * self.kernel.eKxx(self.Xmean).sum()
        eKxx = num_data * var_kernel
        if self.data_type == "iid":
            eKxz_test = self.kernel.eKxz(self.Z, self.Xmean_test, self.Xcov_test)
            eKzxKxz_test = self.kernel.eKzxKxz(self.Z, self.Xmean_test, self.Xcov_test)
            eKxz = th.cat((self.saved_terms["eKxz"], eKxz_test), 0)
            eKzxKxz = self.saved_terms["eKzxKxz"] + eKzxKxz_test
        else:
            print("regressive case not implemented")

        # compute ELBO
        L = self.saved_terms["L"]
        A = trtrs(L, trtrs(L, eKzxKxz).t()) / var_noise.expand_as(L)
        B = A + Variable(th.eye(num_inducing).type(float_type))
        LB = cholesky(B, flag="LB")

        log_2pi = Variable(th.Tensor([np.log(2 * np.pi)]).type(float_type))
        elbo = -dim_output * (
            LB.diag().log().sum() + 0.5 * num_data * (var_noise.log() + log_2pi)
        )
        elbo -= 0.5 * dim_output * (eKxx / var_noise - A.diag().sum())

        if not self.is_large_p:
            # distributed
            # C = Variable(th.zeros(num_inducing, dim_output))
            # for i in xrange(num_data):
            #     C += Psi[i, :].unsqueeze(1).mm(self.Y[i, :].unsqueeze(0))
            C = eKxz.t().mm(Y)
            D = trtrs(LB, trtrs(L, C))
            elbo -= (
                0.5
                * (
                    Y.t().mm(Y) / var_noise.expand(dim_output, dim_output)
                    - D.t().mm(D) / var_noise.pow(2).expand(dim_output, dim_output)
                ).trace()
            )
        else:
            # small n, pre-compute YY'
            # YYT = self.Y.mm(self.Y.t())
            D = trtrs(LB, trtrs(L, eKxz.t()))
            W = Variable(th.eye(num_data).type(float_type)) / var_noise.expand(
                num_data, num_data
            ) - D.t().mm(D) / var_noise.pow(2).expand(num_data, num_data)
            elbo -= 0.5 * (W.mm(self.saved_terms["YYT"])).trace()

        # KL Divergence (KLD) btw the posterior and the prior
        if self.data_type == "iid":
            const_nq = Variable(th.Tensor([num_data * dim_latent]).type(float_type))
            # eqn (3.28) below p57 Damianou's Diss.
            KLD = 0.5 * (
                self.Xmean.pow(2).sum()
                + self.Xcov.transform().sum()
                - self.Xcov.transform().log().sum()
                - const_nq
            )

        elbo -= KLD
        return elbo

    def loss(self):
        if not self.inference:
            return super().loss()
        else:
            return -(self.log_likelihood_inference() + self.log_prior())

    def _pre_compute(self):
        """Pre-computation for the projection

        Fixed terms in test time are manually identified,
        Only iid latent is implemented.
        """
        # Save the fixed terms here
        # self.saved_terms = {}
        if self.observed_dims is not None:
            # select observed dims to compute
            Y = th.cat((self.Y.index_select(1, self.observed_dims), self.Y_test), 0)
            self.saved_terms["YYT"] = Y.mm(Y.t())

        # computes kernel expectations
        if self.data_type == "iid":
            eKxz = self.kernel.eKxz(self.Z, self.Xmean, self.Xcov)
            eKzxKxz = self.kernel.eKzxKxz(self.Z, self.Xmean, self.Xcov)
            self.saved_terms["eKxz"] = eKxz
            self.saved_terms["eKzxKxz"] = eKzxKxz
        else:
            print("regressive case, not implemented")

        Kzz = self.kernel.K(self.Z) + self.jitter.expand(self.Z.size(0)).diag()
        L = cholesky(Kzz, flag="L")
        self.saved_terms["L"] = L

    def project(self, observ_test, observed_dims=None):
        """Infers the latent input corresponding to the new observed data
        The test data can be partially observed.
        # TODO: Currently only the Gaussian approximations, and iid case.
        With recognition model, inference of latent would be faster.

        Args:
            observ_test (numpy.ndarray): Test observed data
            observed_dims (list or np.array, 1D): Observed dimensions of the
                partially observed test data. Must be provided for partially
                observed test case.

        Returns:
            mean and variance of the posterior of Gaussian approximations
        """
        # Set the modes to be the inference mode
        self.inference = True
        if observed_dims is None:
            # Fully observable data
            assert observ_test.shape[1] == self.Y.size(1), (
                "Test data dimension must equal to that of the training "
                "data for the fully observed case, otherwise please "
                "specify the observed dimensions using ``observed_dims``"
            )
        else:
            assert isinstance(observed_dims, (basestring, np.ndarray)), (
                "Type of the list of observed dimensions should be list "
                "or 1d np.array"
            )
            self.observed_dims = Variable(th.LongTensor(observed_dims))
        assert isinstance(observ_test, np.ndarray), "Test data should be " "np.ndarray"
        if observ_test.ndim == 1:
            observ_test = observ_test[None, :]
        # Design choice: do not create a tiny inference model, but reuse the
        # trained model using another function for compute the loss.
        # Add new observation variables to the original class
        self.Y_test = Variable(th.Tensor(observ_test).type(float_type))

        # Freeze the trained parameters
        for param in self.parameters():
            param.requires_grad = False

        # initialize Xmean_test, Xcov_test by searching for the nearest
        # neighbour in the data space
        if observed_dims is None:
            Y_observed = self.Y
        else:
            Y_observed = self.Y.index_select(1, self.observed_dims)

        YYT = self.Y_test.mm(Y_observed.t())
        dist_matrix = (
            -2 * YYT
            + self.Y_test.pow(2).sum(1).expand_as(YYT)
            + Y_observed.t().pow(2).sum(0).expand_as(YYT)
        )
        _, argmin = dist_matrix.min(1)
        argmin = argmin.view(self.Y_test.size(0)).data
        self.Xmean_test = Param(self.Xmean.data[argmin])
        self.Xcov_test = Param(
            self.Xcov.transform().data[argmin], requires_transform=True
        )
        print("GPLVM: Finish preparing the model for projection")
        self._pre_compute()
        print(
            "GPLVM: Done with pre-computation. \nPlease optimize the model"
            " again to obtain the projected latent variables\n"
        )
        # optimize the latent variables
        # Q: how to know the optimization converges? this is slow and painful
        # Thus the model is returned to user for optimization
        # model_project.optimize(method='LBFGS', max_iter=100, verbose=False)
        # return model_project.Xmean, model_project.Xcov
        # self.optimize(method='LBFGS', max_iter=100, verbose=True)
        # use the ``compute_loss_inference`` method during the optimization
        # return self.Xmean_test, self.Xcov_test

    def _predict(self, Xnew_mean, Xnew_var=None, diag=True):
        """Computes the mean and variance of latent function output
        corresponding to the new (uncertain) input

        The new input can be deterministic or uncertain (only Gaussian: mean and
        variance). Returns the predictions over all dimensions (extract the
        needed dimensions for imputation case after getting the returns)

        Args:
             Xnew_mean (np.ndarray): new latent input, it is the deterministic
                input if ``input_var`` is None, otherwise it is the mean of the
                latent posterior, size n_* x q
             Xnew_var (np.ndarray): variance (covariance) of latent posterior,
                iid case, still n_* x q (each row stores the diagonal of cov)

        Returns:
            (Variables): n_* x p, mean of the predicted latent output
            (Variables): covariance of the predicted latent output,
                n_* x p for the deterministic case (share the same covariance),
                or n_* x q x q for the uncertain Gaussian input, iid.

        """
        assert isinstance(Xnew_mean, np.ndarray) and Xnew_mean.shape[
            1
        ] == self.Xmean.size(1), (
            "Input_mean should be numpy.ndarary, and its column dims "
            "should be same as the latent dimensions"
        )
        Xnew_mean = Variable(th.Tensor(Xnew_mean).type(float_type), volatile=True)

        num_inducing = self.Z.size(0)
        beta = 1.0 / self.likelihood.variance.transform()
        # Psi1, Psi2
        eKxz = self.kernel.eKxz(self.Z, self.Xmean, self.Xcov)
        eKzxKxz = self.kernel.eKzxKxz(self.Z, self.Xmean, self.Xcov)
        Kzs = self.kernel.K(self.Z, Xnew_mean)
        Kzz = self.kernel.K(self.Z) + self.jitter.expand(self.Z.size(0)).diag()
        L = cholesky(Kzz, flag="Lkz")
        A = trtrs(L, trtrs(L, eKzxKxz).t()) * beta.expand_as(L)
        B = A + Variable(th.eye(num_inducing).type(float_type))
        Lb = cholesky(B, flag="Lb")
        C = trtrs(L, Kzs)
        D = trtrs(Lb, C)

        if Xnew_var is None:
            # broadcast udpated
            mean = D.t().mm(trtrs(Lb, trtrs(L, eKxz.t().mm(self.Y)))) * beta.expand(
                Xnew_mean.size(0), self.Y.size(1)
            )
            # return full covariance or only the diagonal
            if diag:
                # 1d tensor
                var = (
                    self.kernel.Kdiag(Xnew_mean)
                    - C.pow(2).sum(0).squeeze()
                    + D.pow(2).sum(0).squeeze()
                )
            else:
                var = self.kernel.K(Xnew_mean) - C.t().mm(C) + D.t().mm(D)
        else:
            # uncertain input, assume Gaussian.
            assert (
                isinstance(Xnew_var, np.ndarray) and Xnew_var.shape == Xnew_var.shape
            ), (
                "Uncertain input, inconsistent variance size, "
                "should be numpy ndarray"
            )
            Xnew_var = Param(th.Tensor(Xnew_var).type(float_type))
            Xnew_var.requires_transform = True
            Xnew_var.volatile = True
            # s for star (new input), z for inducing input
            eKsz = self.kernel.eKxz(self.Z, Xnew_mean, Xnew_var)
            # list of n_* expectations w.r.t. each test datum
            eKzsKsz = self.kernel.eKzxKxz(self.Z, Xnew_mean, Xnew_var, sum=False)
            Im = Variable(th.eye(self.Z.size(0)).type(float_type))
            E = trtrs(Lb, trtrs(L, Im))
            EtE = E.t().mm(E)
            F = EtE.mm(eKxz.t().mm(self.Y)) * beta.expand(
                self.Z.size(0), self.Y.size(1)
            )
            mean = eKsz.mm(F)
            Linv = trtrs(L, Im)
            Sigma = Linv.t().mm(Linv) - EtE
            # n x m x m
            # eKzsKsz = eKzsKsz.cat(0).view(Xnew_mean.size(0), *self.Z.size())
            var = []
            if diag:
                ns = Xnew_mean.size(0)
                p = self.Y.size(1)
                # vectorization?
                for i in range(ns):
                    cov = (
                        self.kernel.variance.transform() - Sigma.mm(eKzsKsz[i]).trace()
                    ).expand(p, p) + F.t().mm(
                        eKzsKsz[i]
                        - eKsz[i, :].unsqueeze(0).t().mm(eKsz[i, :].unsqueeze(0))
                    ).mm(
                        F
                    )
                    var.append(cov)
            else:
                # full covariance case, leave for future
                print("multi-output case, future feature")
                var = None
                pass

        return mean, var

    def generate(self, num_samples):
        """Generate new samples from the generative model

        Gaussian mixture model is a good choice for the iid latent.

        .. Note::
            Enforce the posterior of latents to approach the specified prior
            of latents, then samples from the prior, propagates through the
            model.This is the method used in VAEs. But the samples are not
            that good, visually (MNIST).

        .. Note::
            Two ways of drawing samples are different:
            1. Drawing one sample at a time and repeat multiple times ('random')
            2. Drawing multiple samples at a time (smooth)
        """
        # generate new samples from the posterior distributions

    def reconstruct(self, observed_part, observed_dims):
        """Reconstruct the missing dimensions in the test data

        Args:
            observed_part (np.ndarray): Partially observed test data
            observed_dims (slice): indices for the observed dimensions

        Returns:
            missing means and variances of test data
        """
        # 1. optimize q(X_*) - similar to projection
        self.project(observed_part, observed_dims)
        self.optimize(method="LBFGS", max_iter=100)
        # 2. generation / predict
        mean, var = self._predict(self.Xmean_test, self.Xcov_test)
        missing_dims = th.LongTensor(
            np.setdiff1d(range(self.Y.size(1)), self.observed_dims)
        )
        return mean[:, missing_dims], var[:, missing_dims]

    def _forecast(self, time_interval):
        pass
