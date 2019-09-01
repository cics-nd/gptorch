
"""
``gptorch.functions`` contains self-defined functions subclassing
:modile:`torch.autograd.Function` that performs Cholesky decomposition, solving
linear triangular equations, and misc functions.
"""

from __future__ import absolute_import
from .util import as_tensor
import torch
from torch.nn import functional as F
import numpy as np

torch_version = [int(s) for s in torch.__version__.split(".")]
_potri = torch.cholesky_inverse if torch_version >= [1, 1, 0] else torch.potri
    

def jit_op(op, x: torch.Tensor, max_tries: int=10, verbose: bool=False) \
        -> torch.Tensor:
    """
    Attempt a potentially-unstable linear algebra operation on a matrix.
    If it fails, then try adding more and more jitter and try again...
    """
    jitter_size = x.diag().mean()
    try:
        return op(x)
    except Exception as e:
        if verbose:
            print("Op {} failed (initial try)".format(op.__name__))

    for i in range(max_tries):
        try:
            this_jitter = 10.0 ** (-max_tries + i) * torch.eye(*x.shape, 
                dtype=x.dtype)
            return op(x + this_jitter)
        except RuntimeError as e:
            if verbose:
                print("Op {} failed (try {} / {})".format(op.__name__, i + 1, 
                    max_tries))
    raise RuntimeError("Max tries exceeded.")


def cholesky(x: torch.Tensor) -> torch.Tensor:
    return jit_op(torch.cholesky, x)


def cholesky_inverse(x: torch.Tensor, upper=False) -> torch.Tensor:
    """
    Inverse of a matrix based on its Cholesky.
    """
    return _potri(x, upper=upper)


def inverse(x: torch.Tensor) -> torch.Tensor:
    return jit_op(torch.inverse, x)


def lt_log_determinant(L):
    """
    Log-determinant of a triangular matrix

    Args:
        L (Variable): Lower-triangular matrix to take log-determinant of.
    """
    return L.diag().log().sum()


def SoftplusInv(y, lower=1e-6):
    '''Transforms for handling constraints on parameters, e.g. positive variance
    For get the initial value of x, where

    .. math::
        y = \mathrm{Softplus}(x) = \log(1 + e^x)

    SoftplusInv is used to represent the positive constraints of some
    parameters, such as variance.

    Args:
        y (numpy.ndarray or real number): output of softplus,
            value of the parameter value

    Returns:
        the 'free' parameter used in optimization
    '''
    x = torch.log(torch.exp(y - lower) - 1.)
    if y.numpy().any() > 35:
        return y - lower
    else:
        return x


def transform(variable):
    # intent to replace the transform method within the Param class
    assert isinstance(variable, Variable), "Input to this function should be a Variable"
    if variable.requires_transform:
        return F.softplus(variable, threshold=35)
    else:
        return variable


def trtrs(b: torch.Tensor, a: torch.Tensor, lower=True) -> torch.Tensor:
    return torch.triangular_solve(b, a, upper=not lower)[0]
