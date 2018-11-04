
"""
``gptorch.functions`` contains self-defined functions subclassing
:modile:`torch.autograd.Function` that performs Cholesky decomposition, solving
linear triangular equations, and misc functions.
"""

from __future__ import absolute_import
from .util import as_tensor
import torch
from torch.autograd import Function, Variable
from torch.nn import functional as F
import numpy as np


class _TriangularSolve(Function):
    """ Solves linear triangular system A * X = B (B could be multi-columns).
    It uses the following LAPACK function with signature::
        torch.trtrs(B, A, upper=True, transpose=False, unitriangular=False) -> (Tensor, Tensor)
    We do not use transpose and unitriangular here.
    """
    def __init__(self, lower=True):
        super(_TriangularSolve, self).__init__()
        self.upper = not lower
        # self.needs_input_grad = (True, True)

    def forward(self, A, B):
        X = torch.trtrs(B, A, self.upper)[0]
        self.save_for_backward(A, X)
        return X

    def backward(self, grad_output):
        """ Giles, 2008, An extended collection of matrix derivative results
        for forward and reverse mode algorithmic differentiation), sec 2.3.1.

        Args:
            grad_output(sequence of (Tensor, Variable or None)): Gradients
                of the objective function w.r.t. each element of matrix X
                (output of :func:`forward`)

        Returns:
             Tensor: gradient w.r.t. A (triangular matrix)
        """
        grad_A = grad_B = None
        A, X = self.saved_tensors

        if self.needs_input_grad[0]:
            grad_A = - torch.trtrs(grad_output, A, self.upper,
                                   transpose=True,
                                   unitriangular=False)[0].mm(X.t())
            if self.upper:
                grad_A = torch.triu(grad_A)
            else:
                grad_A = torch.tril(grad_A)

        if self.needs_input_grad[1]:
            grad_B = torch.trtrs(grad_output, A, self.upper, transpose=True,
                                 unitriangular=False)[0]

        return grad_A, grad_B
    

def trtrs(tri_matrix, rhs, lower=True):
    """Helper function for the class :class:`_TriangularSolve`,
    you should use this one instead for forward computation.

    .. math::
        AX = B

    Args:
        tri_matrix (Variable): Triangular matrix A
        rhs (Variable): right hand side of linear triangular equation, B
        lower (bool, optional): Default is True, for using the lower part of
            the :attr:`tri_matrix`
    """
    return _TriangularSolve(lower)(tri_matrix, rhs)


class _Cholesky(Function):
    """Implements Cholesky decomposition

    Reference:
        Iain Murray https://github.com/imurray/chol-rev
    """
    def __init__(self, flag='Kuu', rev_algo='symbolic'):
        super(_Cholesky, self).__init__()
        # revere differentiation algorithm, 'blocked' or 'symbolic'
        self.rev_algo = rev_algo
        self.flag = flag

    def forward(self, A):
        """Cholesky decomposition with jittering

        Add jitter to matrix A if A is not positive definite, increase the
        amount of jitter w.r.t number of tries.
        This function uses LAPACK routine::
            torch.potrf(A, upper=True) -> Tensor
        Only enables lower factorization, i.e. A = LL'
        """
        success = False
        max_tries = 10
        i = 0

        while i < max_tries and not success:
            i += 1
            try:
                L = torch.potrf(A, upper=False)
                success = True

            except RuntimeError as e:
                if e.args[0].startswith('Lapack Error in potrf'):
                    print('Warning: Cholesky error for the %d time' % i)
                    A += A.diag().mean(0).expand(A.size(0),).diag() * 1e-6 * \
                         pow(10, i-1)
                    # print(self.flag)
                if i == max_tries:
                    raise e

        self.save_for_backward(L)
        return L

    def backward(self, grad_output):
        """
        Reference:
            eqn (10) & (9) in Iain Murray, 2016, arXiv:1602.07527
        """
        L, = self.saved_tensors
        P = torch.tril(torch.mm(L.t(), grad_output))
        P -= P.diag().diag() / 2.
        S = torch.trtrs(torch.trtrs(P + P.t(), L.t(), upper=True)[0].t(), L.t(),
                        upper=True)[0]
        return S / 2.


def cholesky(A, flag=None):
    """Cholesky decomposition

    .. math::
        A = LL^T

    Args:
        A (Variable or KroneckerProduct): positive definite matrix

    Returns:
        Variable: Lower triangular matrix
    """
    return _Cholesky(flag=flag)(A)


def inverse(input):
    """
    Matrix inverse, will be available in the next stable release

    Args:
        input: the matrix to be inverted
    """
    if isinstance(input, Variable):
        return torch.inverse(input)
    elif isinstance(input, KroneckerProduct):
        return KroneckerProduct.inverse(input)


# def inverse_deprecated(input):
#     """
#     Matrix inverse, will be available in the next stable release
#
#     Args:
#         input: the matrix to be inverted
#     """
#     if isinstance(input, Variable):
#         return _Inverse()(input)
#     elif isinstance(input, KroneckerProduct):
#         return KroneckerProduct.inverse(input)


def lt_log_determinant(L):
    """
    Log-determinant of a triangular matrix

    Args:
        L (Variable or KroneckerProduct):
    """
    if isinstance(L, Variable):
        return L.diag().log().sum()
    elif isinstance(L, KroneckerProduct):
        return np.sum([lt_log_determinant(L.X[i]) * L.n / L.n_i[i]
                       for i in range(len(L.X))])


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
