# Yinhao Zhu
# yzhu10@nd.edu
# April 16, 2017  

"""
Basic model and parameter classes for Gaussian Processes, inheriting from
:class:`torch.nn.Module` and :class:`torch.nn.Parameter` respectively.
"""

from __future__ import absolute_import
from .functions import SoftplusInv
from .util import TensorType

from torch.autograd import Variable, gradcheck
from torch.nn import Module, Parameter
from torch.nn import functional as F
import torch as th
from scipy.optimize import minimize
import numpy as np
from time import time
import warnings
try:
    # Python 2
    from future_builtins import filter
except ImportError:
    # Python 3
    pass

# TODO: samples from the posterior


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # dont do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class Model(Module):
    """
    Customized Model class for all GP objects
    """
    def forward(self):
        return None

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        for name, param in self._parameters.items():
            tmpstr = tmpstr + name + '\n' + \
                     str(param.transform().data)  + '\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + key + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')' + '\n'
        return tmpstr

    # Three functions wrap the scipy.optimize.minimize
    # get_param_array
    # set_parameters
    # loss_and_grad
    def _get_param_array(self):
        """Returns a 1D array by flattening and concatenating all the parameters
        in the model.
        """
        param_array = []
        for param in self.parameters():
            if param.requires_grad:
                param_array.append(param.data.numpy().flatten())

        return np.concatenate(param_array)

    def _set_parameters(self, param_array):
        """Set the parameters from a parameter array in the format of
        the return of get_param_array().
        """
        # assert isinstance(param_array, np.ndarray)
        idx_current = 0
        for param in self.parameters():
            if param.requires_grad:
                idx_next = idx_current + np.prod(param.data.size())
                param_np = np.reshape(param_array[idx_current: idx_next], param.data.numpy().shape)
                idx_current = idx_next
                param.data = TensorType(param_np)

    def _loss_and_grad(self, param_array):
        """
        Computes the loss and gradients for the parameters with values
        ```param_array```.

        1) Take param_array and expand it into the model
        2) Compute the loss and grad
        3) Put the results into numpy data structures

        Note: Any grad entries that are infinite are replaced with zeroes.

        Args:
            param_array (np.ndarray): the state x

        Returns:
            f(x) (np.float64), the objective function values
            g(x) (np.ndarray of np.float64's), the gradient
        """

        # 1) set parameters of the model from a 1D param_array
        self._set_parameters(param_array)
        # print(self)
        for name, param in self.named_parameters():
            # param.grad will be None if either (1) this is the first time we've
            # visited or (2) this param doesn't require a gradient
            if param.grad is not None:
                param.grad.data.zero_()

        # 2) Compute
        loss = self.compute_loss()
        loss.backward()

        # 3) Return as the proper numpy types
        grad = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                grad.append(param.grad.data.numpy().flatten())
            # if name in ['Z']:
            #     print('Z: %s' % param.data)
            #     print('grad of Z: %s' % param.grad)
        print('loss: %s' % loss.data.numpy())
        grad = np.concatenate(grad)
        grad_isfinite = np.isfinite(grad)
        ## scipy.optimizer.minimize.L-BFGS-B expects double precision (Fortran)
        if np.all(grad_isfinite):
            return loss.data.numpy().astype(np.float64), grad.astype(np.float64)
        else:
            # self._previous_x = x  # store the last known good value
            print("Warning: inf or nan in gradient: replacing with zeros")
            return loss.data.numpy().astype(np.float64), \
                   np.where(grad_isfinite, grad, 0.).astype(np.float64)

    # Functions for using gradcheck on your model.
    # Use self.loss as the function to be grad-checked, and provide the inputs
    # with self.extract_params
    def extract_params(self):
        """
        Returns:
            (Tuple of Variables)
        """
        return tuple([x for x in self.parameters()])

    def expand_params(self, *args):
        """
        Args:
            args (tuple of Variables): (Get from self.extract_params())
        """
        for arg, (param_name, param) in zip(args, self.named_parameters()):
            if isinstance(arg, Param):
                param.data = arg.data
            elif isinstance(arg, np.ndarray):
                raise NotImplementedError(
                    "Unresolved issues with expanding numpy arrays")
                # param.data = th.Tensor(arg).type(float_type)

    def loss(self, *args):
        """
        Loss, given args to be expanded into the model.

        Args:
            args (pointer to a tuple of Variables):
            (Get from self.extract_params())
        """
        self.expand_params(*args)
        return self.compute_loss()

    def gradcheck(self, eps=1e-6, atol=1e-5, rtol=1e-3, verbose=False):
        """
        This performs a gradcheck on your model's self.compute_loss() function

        See torch.autograd.gradcheck for more on eps, atol, and rtol.

        Args:
            eps (float): Finite difference step size
            atol (float): absolute tolerance (takes over with grads near zero)
            rtol (float): relative tolerance (takes over with large grads)
            verbose (bool): Output more
        Returns:
             (bool) whether the gradcheck passed (true = good)
        """
        if verbose:
            warnings.warn("Verbose not yet figured out")
        return gradcheck(self.loss, self.extract_params(), eps=eps, atol=atol,
                         rtol=rtol)


class Param(Parameter):
    """
    Customized Parameter class
    Add constraints (using transform) to parameters.
    Currently only support positive constraints, e.g. for variance

    prior is an instance of the gptorch.Prior class.
    """
    def __new__(cls, data=None, requires_grad=True, requires_transform=False,
                prior=None):
        if requires_transform:
            data = Param._transform_log(data, forward=False)
        return super(Parameter, cls).__new__(cls, data,
                                             requires_grad=requires_grad)

    def __init__(self, data, requires_grad=True, requires_transform=False):
        self.requires_transform = requires_transform
        self.prior = None
        super(Param, self).__init__(data, requires_grad=requires_grad)

    def transform(self):
        # Avoid in-place operation for Variable, using clone method  ???
        if self.requires_transform:
            return self._transform_log(self.clone(), forward=True)
        else:
            return self

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()

    @staticmethod
    def _transform_log(x, forward):
        if forward:
            return th.exp(x)
        else:
            return th.log(x)

    @staticmethod
    def _transform_softplus(x, forward):
        if forward:
            return F.softplus(x, threshold=35)
        else:
            return SoftplusInv(x)


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
            mean_function (gptorch.MeanFunction):
            name (string): name of this model
        """
        super().__init__()
        self.kernel = kernel
        self.likelihood = likelihood
        # TODO: reworking mean functions as parameterized classes instead of
        # design & weight matrices.
        assert mean_function is None, "Mean functions not supported"
        self.mean_function = mean_function

        assert(type(y) is np.ndarray or
               type(y) is Variable), "observations must be either " \
                                                "a numpy array or a torch " \
                                                "Variable."
        if isinstance(y, np.ndarray):
            y = Variable(TensorType(y),
                                    requires_grad=False)

        assert (type(x) is np.ndarray or
                type(x) is Variable), \
            "x must be either a numpy array or a torch Variable."
        if isinstance(x, np.ndarray):
            # x is a data matrix; each row represents one instance
            x = Variable(TensorType(x), requires_grad=False)
        self.Y = y
        self.X = x
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
            self.optimizer = th.optim.SGD(parameters, lr=0.05, momentum=0.9)
        elif method == 'Adam':
            self.optimizer = th.optim.Adam(parameters, lr=0.001,
                                           betas=(0.9, 0.999), eps=1e-06, weight_decay=0.0)
        elif method == 'LBFGS':
            if learning_rate is None:
                learning_rate = 1.0
            self.optimizer = th.optim.LBFGS(parameters, lr=learning_rate,
                                            max_iter=5, max_eval=None,
                                            tolerance_grad=1e-05,
                                            tolerance_change=1e-09,
                                            history_size=50,
                                            line_search_fn=None)
        elif method == 'Adadelta':
            self.optimizer = th.optim.Adadelta(
                parameters, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0.00001)
        elif method == 'Adagrad':
            self.optimizer = th.optim.Adagrad(
                parameters, lr=0.01, lr_decay=0, weight_decay=0)
        elif method == 'Adamax':
            self.optimizer = th.optim.Adamax(
                parameters, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif method == 'ASGD':
            self.optimizer = th.optim.ASGD(
                parameters, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        elif method == 'RMSprop':
            self.optimizer = th.optim.RMSprop(
                parameters, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0.00, momentum=0.01, centered=False)
        elif method == 'Rprop':
            self.optimizer = th.optim.Rprop(
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
        chol_s = th.potrf(sigma, upper=False)
        samp = mu + th.stack([th.mm(chol_s, Variable(TensorType(r)))
                              for r in np.random.randn(n_samples, *mu.size())])
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
            return np.power(y_pred_mean - y_test, 2).sum() / y_test.shape[0] / y_test.var()
        elif metric == 'RMSE':
            return np.sqrt(np.power(y_pred_mean - y_test, 2).sum() / y_test.shape[0])
        elif metric == 'MSLL':
            # single output dimension
            # predictions for each independent output dimension are the same
            # fitting training data with trivial Guassian
            y_train = self.Y.data.numpy()
            m0 = y_train.mean()
            S0 = y_train.var()
            msll = 0.5 * np.mean(np.log(2 * np.pi * y_pred_var) + np.power(y_pred_mean - y_test, 2) / y_pred_var) - \
                   0.5 * np.mean(np.log(2 * np.pi * S0) + np.power(y_test - m0, 2) / S0)
            # 0.5 * (y_test.shape[0] * np.log(2 * np.pi * S0) + np.sum(np.power(y_test - m0, 2) / S0))
            return msll
        elif metric == 'NLML':
            return self.compute_loss().data.numpy()
        else:
            raise Exception('No such metric are supported currently, select one of the following:'
                            'SMSE, RSME, MSLL, NLML.')
