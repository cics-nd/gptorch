# File: base.py
# File Created: Saturday, 13th July 2019 1:02:55 pm
# Author: Steven Atkinson (steven@atkinson.mn)

"""
base.py: The core GP model (GP+likelihood)
"""

import numpy as np
import torch
from time import time
from scipy.optimize import minimize

from .. import likelihoods
from ..functions import cholesky
from ..mean_functions import Zero
from ..model import Model
from ..util import torch_dtype

torch.set_default_dtype(torch_dtype)


def input_as_tensor(predict_func):
    """
    Decorator for prediction funtions to ensure that inputs are torch.Tensor
    objects before passing into GPModel._predict() methods.

    :param predict_func: The public predict funciton to be wrapped
    """

    def predict(obj, input_new, *args, **kwargs):
        from_numpy = isinstance(input_new, np.ndarray)
        if from_numpy:
            input_new = torch.Tensor(input_new)
        out = predict_func(obj, input_new, *args, **kwargs)
        if from_numpy:
            if isinstance(out, torch.Tensor):
                out = out.detach().cpu().numpy()
            elif isinstance(out, tuple):
                out = tuple([o.detach().cpu().numpy() for o in out])
            else:
                raise NotImplementedError("Unhandled output type")
        return out

    return predict


class GPModel(Model):
    """
    The base class for GP models
    """

    def __init__(self, x, y, kernel, likelihood, mean_function, name="gp"):
        """
        For unsupervised case, x is optional ...
        Args:
            y (ndarray): Y, N x q
            x (ndarray): X, N x p
            kernel (gptorch.Kernel):
            likelihood (gptorch.Likelihood):
            mean_function (gptorch.MeanFunction):
            name (string): name of this model
        """

        super().__init__()
        self.kernel = kernel
        self.likelihood = likelihood if likelihood is not None else \
            GPModel._init_gaussian_likelihood(y)
        self.mean_function = mean_function if mean_function is not None else \
            Zero(y.shape[1])

        allowed_data_types = (np.ndarray, torch.Tensor)
        assert type(x) in allowed_data_types, "x must be one of {}".format(
            allowed_data_types
        )
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        assert type(y) in allowed_data_types, "y must be one of {}".format(
            allowed_data_types
        )
        if isinstance(y, np.ndarray):
            y = torch.Tensor(y)
        x.requires_grad_(False)
        y.requires_grad_(False)
        self.X, self.Y = x, y

        self.__class__.__name__ = name

    @property
    def num_data(self):
        return self.Y.shape[0]

    @property
    def input_dimension(self):
        return self.X.shape[1]

    @property
    def output_dimension(self):
        return self.Y.shape[1]

    def compute_loss(self, x=None, y=None):
        """
        Defines computation graph upon every call - PyTorch
        This function must be implemented by all subclasses.

        :param x: provide the training inputs on which the loss will be computed.
            If not provided, default to self.X,
        :param y: provide the training outputs on which the loss will be computed.
            If not provided, default to self.Y
        """
        raise NotImplementedError

    # def forward(self, *data):
    #     # Builds computation graph for an objective, e.g. log marginal likelihood
    #     # Responds to the NotImplementedError in super class: torch.nn.Module
    #     # data usually includes observations and input
    #     return self.compute_likelihood(*data)

    @staticmethod
    def _init_gaussian_likelihood(y) -> likelihoods.Gaussian:
        """
        A handy heuristic for initializing Gaussian likelihoods for models: make
        the standard deviation roughly 3% of the total output variance.

        :param y: Outputs.  Can be either torch.Tensor or np.ndarray
        """
        return likelihoods.Gaussian(variance=0.001 * y.var())

    def optimize(self, method='Adam', max_iter=2000, verbose=True,
            learning_rate=None):
        """
        Optimizes the model by minimizing the loss (from :method:) w.r.t.
        model parameters.
        Args:
            method (torch.optim.Optimizer, optional): Optimizer in PyTorch
                (maybe add scipy optimizer in the future), default is `Adam`.
            max_iter (int): Max iterations, default 2000.
            verbose (bool, optional): Shows more details on optimization
                process if True.
        Todo:
            Add stochastic optimization, such as mini-batch.
        Returns:
            (np.array, value):
                losses: losses over optimization steps, (max_iter, )
                time: time taken approximately
        """
        parameters = filter(lambda p: p.requires_grad, self.parameters())

        default_learning_rates = {
            "SGD": 0.001,
            "Adam": 0.01,
            "LBFGS": 1.0,
            "Adadelta": 1.0,
            "Adagrad": 0.01,
            "Adamax": 0.002,
            "ASGD": 0.01,
            "RMSprop": 0.01,
            "Rprop": 0.01
        }
        if learning_rate is None and method in default_learning_rates:
            learning_rate = default_learning_rates[method]
        if method == "SGD":
            # GPs seem to benefit from a more aggressive learning rate than the
            # usual "0.001 or so" that NNs like.
            learning_rate = learning_rate if learning_rate is not None else 0.01
            self.optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
        elif method == "Adam":
            learning_rate = learning_rate if learning_rate is not None else 0.01
            self.optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        elif method == "LBFGS":
            if learning_rate is None:
                learning_rate = 1.0
            self.optimizer = torch.optim.LBFGS(
                parameters,
                lr=learning_rate,
                max_iter=5,
                max_eval=None,
                tolerance_grad=1e-05,
                tolerance_change=1e-09,
                history_size=50,
                line_search_fn=None,
            )
        elif method == "Adadelta":
            self.optimizer = torch.optim.Adadelta(
                parameters, lr=learning_rate, rho=0.9, eps=1e-06, 
                weight_decay=0.00001
            )
        elif method == "Adagrad":
            self.optimizer = torch.optim.Adagrad(
                parameters, lr=learning_rate, lr_decay=0, weight_decay=0
            )
        elif method == "Adamax":
            self.optimizer = torch.optim.Adamax(
                parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
            )
        elif method == "ASGD":
            self.optimizer = torch.optim.ASGD(
                parameters,
                lr=learning_rate,
                lambd=0.0001,
                alpha=0.75,
                t0=1000000.0,
                weight_decay=0,
            )
        elif method == "RMSprop":
            self.optimizer = torch.optim.RMSprop(
                parameters,
                lr=learning_rate,
                alpha=0.99,
                eps=1e-08,
                weight_decay=0.00,
                momentum=0.01,
                centered=False,
            )
        elif method == "Rprop":
            self.optimizer = torch.optim.Rprop(
                parameters, lr=learning_rate, etas=(0.5, 1.2), step_sizes=(1e-06, 50)
            )
        # scipy.optimize.minimize
        # suggest to use L-BFGS-B, BFGS
        elif method in [
            "CG",
            "BFGS",
            "Newton-CG",
            "Nelder-Mead",
            "Powell",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "SLSQP",
            "dogleg",
            "trust-ncg",
        ]:
            print("Scipy.optimize.minimize...")
            return self._optimize_scipy(method=method, maxiter=max_iter, disp=verbose)

        else:
            raise ValueError(
                "Optimizer %s is not found. Please choose one of the"
                "following optimizers supported in PyTorch:"
                "Adadelt, Adagrad, Adam, Adamax, ASGD, LBFGS, "
                "RMSprop, Rprop, SGD, LBFGS. Or the optimizers "
                "supported scipy.optimize.minminze: BFGS, L-BFGS-B,"
                "CG, Newton-CG, Nelder-Mead, Powell, TNC, COBYLA,"
                "SLSQP, dogleg, trust-ncg, etc." % method
            )

        losses = np.zeros(max_iter)
        tic = time()

        print("{}: Start optimizing via {}".format(self.__class__.__name__, method))
        if verbose:
            if not method == "LBFGS":
                for idx in range(max_iter):
                    self.optimizer.zero_grad()
                    # forward
                    loss = self.compute_loss()
                    # backward
                    loss.backward()
                    self.optimizer.step()
                    losses[idx] = loss.data.numpy()
                    print("Iter: %d\tLoss: %s" % (idx, loss.data.numpy()))

            else:
                for idx in range(max_iter):

                    def closure():
                        self.optimizer.zero_grad()
                        loss = self.compute_loss()
                        loss.backward()
                        return loss

                    loss = self.optimizer.step(closure)
                    losses[idx] = loss.data.numpy()
                    print("Iter: %d\tLoss: %s" % (idx, loss.data.numpy()))
        else:
            if not method == "LBFGS":
                for idx in range(max_iter):
                    self.optimizer.zero_grad()
                    # forward
                    loss = self.compute_loss()
                    # backward
                    loss.backward()
                    self.optimizer.step()
                    losses[idx] = loss.data.numpy()
                    if idx % 20 == 0:
                        print("Iter: %d\tLoss: %s" % (idx, loss.data.numpy()))
            else:
                for idx in range(max_iter):

                    def closure():
                        self.optimizer.zero_grad()
                        loss = self.compute_loss()
                        loss.backward()
                        return loss

                    loss = self.optimizer.step(closure)
                    if isinstance(loss, float):  # Converged!
                        losses[idx] = loss
                        losses = losses[0 : idx + 1]
                        break
                    else:
                        losses[idx] = loss.data.numpy()
                    if idx % 20 == 0:
                        print("Iter: %d\tLoss: %s" % (idx, losses[idx]))
        t = time() - tic
        print("Optimization time taken: %s s" % t)
        print("Optimization method: %s" % str(self.optimizer))
        if len(losses) == max_iter:
            print("Optimization terminated by reaching the maximum iterations")
        else:
            print("Optimization terminated by getting below the tolerant error")

        return losses, t

    def _optimize_scipy(
        self, method="L-BFGS-B", tol=None, callback=None, maxiter=1000, disp=True
    ):
        """
        Wrapper of scipy.optimize.minimize
        Args:
             method (string): Optimization method
             maxiter (int, optional): Maximum number of iterations to perform,
                default is 1000
             disp (bool, optional): Set to True to print convergence messages.
        """
        options = dict(disp=disp, maxiter=maxiter)

        result = minimize(
            fun=self._loss_and_grad,
            x0=self._get_param_array(),
            method=method,
            jac=True,
            tol=tol,
            callback=callback,
            options=options,
        )
        return result

    def _predict(self, input_new: torch.Tensor, diag=True):
        """
        Predict hte latent output function at input_new.

        If diag=True, then the outputs are the predictive mean and variance, 
        both of shape [n x dy].
        If diag=False, then we predict the full covariance [n x n]; the mean is 
        the same.

        :return: (torch.Tensor, torch.Tensor)
        """
        # diag: is a flag indicates whether only returns the diagonal of
        # predictive variance at input_new
        # :param input_new: np.ndarray
        raise NotImplementedError()

    @input_as_tensor
    def predict_f(self, input_new, diag=True, **kwargs):
        """
        Computes the mean and variance of the latent function at input_new
        return the diagonal of the cov matrix
        Args:
            input_new (numpy.ndarray or torch.Tensor)
        """
        return self._predict(input_new, diag=diag, **kwargs)

    @input_as_tensor
    def predict_y(self, input_new, diag=True, **kwargs):
        """
        Computes the mean and variance of observations at new inputs
        Args:
            input_new (numpy.ndarray)
        """
        mean_f, cov_f = self._predict(input_new, diag=diag, **kwargs)

        if diag:
            return self.likelihood.predict_mean_variance(mean_f, cov_f)
        else:
            return self.likelihood.predict_mean_covariance(mean_f, cov_f)

    @input_as_tensor
    def predict_f_samples(self, input_new, n_samples=1, **kwargs):
        """
        Return [n_samp x n_test x d_y] matrix of samples
        :param input_new:
        :param n_samples:
        :return:
        """
        mu, sigma = self.predict_f(input_new, diag=False, **kwargs)
        chol_s = cholesky(sigma)
        samp = mu + chol_s[None, :, :] @ torch.randn(n_samples, *mu.shape)
        return samp

    @input_as_tensor
    def predict_y_samples(self, input_new, n_samples=1, **kwargs):
        """
        Return [n_samp x n_test x d_y] matrix of samples
        :param input_new:
        :param n_samples:
        :return:
        """
        mu, sigma = self.predict_y(input_new, diag=False, **kwargs)
        chol_s = cholesky(sigma)
        samp = mu + chol_s[None, :, :] @ torch.randn(n_samples, *mu.shape)
        return samp

    # TODO: need more thought on this interface
    # convert the np operations into tensor ops
    def evaluate(self, x_test, y_test, metric="NLML"):
        """
        Evaluate the model using various metrics, including:
        - SMSE: Standardized Mean Squared Error
        - RMSE: Rooted Mean Squared Error
        - MSLL: Mean Standardized Log Loss
        - NLML: Negative Log Marginal Likelihood
        Args:
            x_test (numpy.ndarray): test input
            y_test (numpy.ndarray): test observations
            metric (string, optional): name for the metric, chosen from the list
                above.
        Returns:
            value of the metric
        """

        y_pred_mean, y_pred_var = self.predict_y(x_test)
        y_pred_mean, y_pred_var = y_pred_mean.data.numpy(), y_pred_var.data.numpy()

        if metric == "SMSE":
            return (
                np.power(y_pred_mean - y_test, 2).sum() / y_test.shape[0] / y_test.var()
            )
        elif metric == "RMSE":
            return np.sqrt(np.power(y_pred_mean - y_test, 2).sum() / y_test.shape[0])
        elif metric == "MSLL":
            # single output dimension
            # predictions for each independent output dimension are the same
            # fitting training data with trivial Guassian
            y_train = self.Y.data.numpy()
            m0 = y_train.mean()
            S0 = y_train.var()
            msll = 0.5 * np.mean(
                np.log(2 * np.pi * y_pred_var)
                + np.power(y_pred_mean - y_test, 2) / y_pred_var
            ) - 0.5 * np.mean(np.log(2 * np.pi * S0) + np.power(y_test - m0, 2) / S0)
            # 0.5 * (y_test.shape[0] * np.log(2 * np.pi * S0) + \
            # np.sum(np.power(y_test - m0, 2) / S0))
            return msll
        elif metric == "NLML":
            return self.compute_loss().data.numpy()
        else:
            raise Exception(
                "No such metric are supported currently, "
                + "select one of the following: SMSE, RSME, MSLL, NLML."
            )