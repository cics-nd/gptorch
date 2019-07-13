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

from ..functions import cholesky
from ..mean_functions import Zero
from ..model import Model
from ..util import torch_dtype

torch.set_default_dtype(torch_dtype)


class GPModel(Model):
    """
    The base class for GP models
    """
    def __init__(self, y, x, kernel, likelihood, mean_function, 
                 name='gp'):
        """
        For unsupervised case, x is optional ...

        Args:
            y (ndarray): Y, N x q
            x (ndarray): X, N x p
            kernel (gptorch.Kernel):
            likelihood (gptorch.Likelihood):
            mean_function (torch.nn.Module):
            name (string): name of this model
        """
        super().__init__()
        self.kernel = kernel
        self.likelihood = likelihood
        # TODO: reworking mean functions as parameterized classes instead of
        # design & weight matrices.
        self.mean_function = mean_function if mean_function is not None else \
            Zero(y.shape[1])

        allowed_data_types = (np.ndarray, torch.Tensor)
        assert type(x) in allowed_data_types, \
            "x must be one of {}".format(allowed_data_types)
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        assert type(y) in allowed_data_types, \
            "y must be one of {}".format(allowed_data_types)
        if isinstance(y, np.ndarray):
            y = torch.Tensor(y)
        x.requires_grad_(False)
        y.requires_grad_(False)
        self.X, self.Y = x, y
        self.__class__.__name__ = name

    def compute_loss(self):
        """
        Defines computation graph upon every call - PyTorch

        This function must be implemented by all subclasses.
        """
        raise NotImplementedError

    # def forward(self, *data):
    #     # Builds computation graph for an objective, e.g. log marginal likelihood
    #     # Responds to the NotImplementedError in super class: torch.nn.Module
    #     # data usually includes observations and input
    #     return self.compute_likelihood(*data)

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

        if method == 'SGD':
            self.optimizer = torch.optim.SGD(parameters, lr=0.05, momentum=0.9)
        elif method == 'Adam':
            self.optimizer = torch.optim.Adam(parameters, lr=0.001,
                                           betas=(0.9, 0.999), eps=1e-06, weight_decay=0.0)
        elif method == 'LBFGS':
            if learning_rate is None:
                learning_rate = 1.0
            self.optimizer = torch.optim.LBFGS(parameters, lr=learning_rate,
                                            max_iter=5, max_eval=None,
                                            tolerance_grad=1e-05,
                                            tolerance_change=1e-09,
                                            history_size=50,
                                            line_search_fn=None)
        elif method == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(
                parameters, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0.00001)
        elif method == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(
                parameters, lr=0.01, lr_decay=0, weight_decay=0)
        elif method == 'Adamax':
            self.optimizer = torch.optim.Adamax(
                parameters, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif method == 'ASGD':
            self.optimizer = torch.optim.ASGD(
                parameters, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        elif method == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(
                parameters, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0.00, momentum=0.01, centered=False)
        elif method == 'Rprop':
            self.optimizer = torch.optim.Rprop(
                parameters, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        # scipy.optimize.minimize
        # suggest to use L-BFGS-B, BFGS
        elif method in ['CG', 'BFGS', 'Newton-CG', 'Nelder-Mead', 'Powell',
                        'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg',
                        'trust-ncg']:
            print('Scipy.optimize.minimize...')
            return self._optimize_scipy(method=method, maxiter=max_iter,
                                        disp=verbose)

        else:
            raise Exception('Optimizer %s is not found. Please choose one of the'
                            'following optimizers supported in PyTorch:'
                            'Adadelt, Adagrad, Adam, Adamax, ASGD, LBFGS, '
                            'RMSprop, Rprop, SGD, LBFGS. Or the optimizers '
                            'supported scipy.optimize.minminze: BFGS, L-BFGS-B,'
                            'CG, Newton-CG, Nelder-Mead, Powell, TNC, COBYLA,'
                            'SLSQP, dogleg, trust-ncg, etc.' % method)

        losses = np.zeros(max_iter)
        tic = time()

        print('{}: Start optimizing via {}'.format(self.__class__.__name__, method))
        if verbose:
            if not method == 'LBFGS':
                for idx in range(max_iter):
                    self.optimizer.zero_grad()
                    # forward
                    loss = self.compute_loss()
                    # backward
                    loss.backward()
                    self.optimizer.step()
                    losses[idx] = loss.data.numpy()
                    print('Iter: %d\tLoss: %s' % (idx, loss.data.numpy()))

            else:
                for idx in range(max_iter):
                    def closure():
                        self.optimizer.zero_grad()
                        loss = self.compute_loss()
                        loss.backward()
                        return loss
                    loss = self.optimizer.step(closure)
                    losses[idx] = loss.data.numpy()
                    print('Iter: %d\tLoss: %s' % (idx, loss.data.numpy()))
        else:
            if not method == 'LBFGS':
                for idx in range(max_iter):
                    self.optimizer.zero_grad()
                    # forward
                    loss = self.compute_loss()
                    # backward
                    loss.backward()
                    self.optimizer.step()
                    losses[idx] = loss.data.numpy()
                    if idx % 20 == 0:
                        print('Iter: %d\tLoss: %s' % (idx, loss.data.numpy()))
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
                        losses = losses[0:idx+1]
                        break
                    else:
                        losses[idx] = loss.data.numpy()
                    if idx % 20 == 0:
                        print('Iter: %d\tLoss: %s' % (idx, losses[idx]))
        t = time() - tic
        print('Optimization time taken: %s s' % t)
        print('Optimization method: %s' % str(self.optimizer))
        if len(losses) == max_iter:
            print('Optimization terminated by reaching the maximum iterations')
        else:
            print('Optimization terminated by getting below the tolerant error')

        return losses, t

    def _optimize_scipy(self, method='L-BFGS-B', tol=None, callback=None,
                        maxiter=1000, disp=True):
        """
        Wrapper of scipy.optimize.minimize

        Args:
             method (string): Optimization method
             maxiter (int, optional): Maximum number of iterations to perform,
                default is 1000
             disp (bool, optional): Set to True to print convergence messages.
        """
        options = dict(disp=disp, maxiter=maxiter)

        result = minimize(fun=self._loss_and_grad,
                          x0=self._get_param_array(),
                          method=method,
                          jac=True,
                          tol=tol,
                          callback=callback,
                          options=options)
        return result

    def _predict(self, input_new, diag=True):
        # diag: is a flag indicates whether only returns the diagonal of
        # predictive variance at input_new
        # :param input_new: np.ndarray
        raise NotImplementedError

    def predict_f(self, input_new):
        """
        Computes the mean and variance of the latent function at input_new
        return the diagonal of the cov matrix

        Args:
            input_new (numpy.ndarray)
        """
        return self._predict(input_new, diag=True)

    def _predict_f_cov_matrix(self, input_new):
        # return the full predictive cov matrix of latent function at new inputs
        return self._predict(input_new, diag=False)

    def predict_y(self, input_new, diag=True):
        """
        Computes the mean and variance of observations at new inputs

        Args:
            input_new (numpy.ndarray)
        """
        mean_f, cov_f = self._predict(input_new, diag=diag)
        # print(mean_f, var_f)
        if diag:
            return self.likelihood.predict_mean_variance(mean_f, cov_f)
        else:
            return self.likelihood.predict_mean_covariance(mean_f, cov_f)

    def predict_y_samples(self, input_new, n_samples=1):
        """
        Return [n_samp x n_test x d_y] matrix of samples
        :param input_new:
        :param n_samples:
        :return:
        """
        mu, sigma = self.predict_y(input_new, False)
        chol_s = cholesky(sigma)
        # Batch matmul uses leading indices as batch indices
        samp = mu + chol_s @ torch.randn(n_samples, *mu.shape)
        return samp

    # TODO: need more thought on this interface
    # convert the np operations into tensor ops
    def evaluate(self, x_test, y_test, metric='NLML'):
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

        if metric == 'SMSE':
            return np.power(y_pred_mean - y_test, 2).sum() / y_test.shape[0] / \
                y_test.var()
        elif metric == 'RMSE':
            return np.sqrt(np.power(y_pred_mean - y_test, 2).sum() / \
                y_test.shape[0])
        elif metric == 'MSLL':
            # single output dimension
            # predictions for each independent output dimension are the same
            # fitting training data with trivial Guassian
            y_train = self.Y.data.numpy()
            m0 = y_train.mean()
            S0 = y_train.var()
            msll = 0.5 * np.mean(np.log(2 * np.pi * y_pred_var) + \
                np.power(y_pred_mean - y_test, 2) / y_pred_var) - \
                0.5 * np.mean(np.log(2 * np.pi * S0) + \
                np.power(y_test - m0, 2) / S0)
            # 0.5 * (y_test.shape[0] * np.log(2 * np.pi * S0) + \
            # np.sum(np.power(y_test - m0, 2) / S0))
            return msll
        elif metric == 'NLML':
            return self.compute_loss().data.numpy()
        else:
            raise Exception('No such metric are supported currently, ' + 
                'select one of the following: SMSE, RSME, MSLL, NLML.')
