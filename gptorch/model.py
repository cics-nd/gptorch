# Yinhao Zhu
# yzhu10@nd.edu
# April 16, 2017

"""
Basic model and parameter classes for Gaussian Processes, inheriting from
:class:`torch.nn.Module` and :class:`torch.nn.Parameter` respectively.
"""

from torch.autograd import gradcheck
import torch
import numpy as np
from time import time
from warnings import warn

from .functions import cholesky
from .param import Param
from .util import TensorType


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # dont do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class Model(torch.nn.Module):
    """
    Customized Model class for all GP objects
    """

    def forward(self):
        return None

    def __repr__(self):
        tmpstr = self.__class__.__name__ + " (\n"
        for name, param in self._parameters.items():
            tmpstr = tmpstr + name + "\n" + str(param.transform().data) + "\n"
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + "  (" + key + "): " + modstr + "\n"
        tmpstr = tmpstr + ")" + "\n"
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
                param_array.append(param.detach().cpu().numpy().flatten())

        return np.concatenate(param_array)

    def _set_parameters(self, param_array):
        """Set the parameters from a parameter array in the format of
        the return of get_param_array().
        """
        # assert isinstance(param_array, np.ndarray)
        idx_current = 0
        for param in self.parameters():
            if param.requires_grad:
                idx_next = idx_current + param.numel()
                param_in = TensorType(
                    np.reshape(param_array[idx_current:idx_next], param.shape)
                )
                if param.is_cuda:
                    param_in = param_in.cuda()
                param.data = param_in
                idx_current = idx_next

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
        loss = self.loss()
        loss.backward()

        # 3) Return as the proper numpy types
        grad = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                grad.append(param.grad.cpu().numpy().flatten())
        print("loss: %s" % loss.item())
        grad = np.concatenate(grad)
        grad_isfinite = np.isfinite(grad)
        ## scipy.optimizer.minimize.L-BFGS-B expects double precision (Fortran)
        if np.all(grad_isfinite):
            return float(loss.item()), grad.astype(np.float64)
        else:
            # self._previous_x = x  # store the last known good value
            print("Warning: inf or nan in gradient: replacing with zeros")
            return (
                loss.item(),
                np.where(grad_isfinite, grad, 0.0).astype(np.float64),
            )

    # Functions for using gradcheck on your model.
    # Use self.loss as the function to be grad-checked, and provide the inputs
    # with self.extract_params
    def extract_params(self):
        """
        Returns:
            (Tuple of TensorTypes)
        """
        return tuple([x for x in self.parameters()])

    def expand_params(self, *args):
        """
        Args:
            args (tuple of TensorTypes): (Get from self.extract_params())
        """
        for arg, (param_name, param) in zip(args, self.named_parameters()):
            if isinstance(arg, Param):
                param.data = arg.data
            elif isinstance(arg, np.ndarray):
                raise NotImplementedError(
                    "Unresolved issues with expanding numpy arrays"
                )

    def log_prior(self):
        """
        Compute the log prior of the model

        Searches through all of the model's parameters and evaluates the 
        log-probability on everything that has a prior.

        :return: (TensorType) The log-prior
        """

        log_prior = 0.0
        for param in self.parameters():
            if hasattr(param, "prior") and param.prior is not None:
                if hasattr(param, "transform") and param.transform is not None:
                    val = param.transform()
                else:
                    val = param.data
                log_prior += param.prior.log_prob(val).sum()

        return log_prior

    def loss(self, *loss_args, params=None, **loss_kwargs):
        """
        Loss, given args to be expanded into the model.

        :param params: The state of the module.  See `.extract_params()`. If 
        provided, the model state is updated before evaluating the loss.
        :type params: tuple of gptorch.util.TensorType objects
        :param loss_args: arguments to pass to the child class's `._loss()` 
        method.
        :type loss_args: tuple
        :param loss_kwargs: Keyword arguments to pass to the child class's 
        `._loss()` method.
        :type loss_kwargs: dict
        """

        if params is not None:
            self.expand_params(*params)
        
        return self._loss(*loss_args, **loss_kwargs)

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
            warn("Verbose not yet figured out")
        return gradcheck(
            self.loss, self.extract_params(), eps=eps, atol=atol, rtol=rtol
        )

    def _loss(self, *args, **kwargs):
        raise NotImplementedError("Implement loss function")
