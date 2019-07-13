# File: mean_functions.py
# Author: Yinhao Zhu (yzhu10@nd.edu)

from torch.autograd import Variable
import torch as th
from numpy.polynomial.hermite import hermval
import numpy as np

from .util import as_variable, torch_dtype, TensorType
from .model import Param  # FIXME circular import on model.Model!


class MeanFunction(th.nn.Module):
    """
    The base mean function class.
    To implement a mean function, write the __call__ method. This takes a
    tensor X and returns a tensor m(X). In accordance with the GPflow/GPtorch
    standard, each row of X represents one datum, and each row of Y is computed
    independently for each row of X.

    MeanFunction classes can have parameters, see the Linear class for an
    example.
    """
    def __init__(self):
        super(MeanFunction, self).__init__()

    def __call__(self, X):
        raise NotImplementedError("Implement the __call__\
                                  method for this mean function")

    def __add__(self, other):
        return Additive(self, other)

    def __mul__(self, other):
        return Product(self, other)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n\n'
        for name, param in self._parameters.items():
            tmpstr = tmpstr + name + str(param.data) + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr


class Zero(MeanFunction):
    def __init__(self, dy):
        super().__init__()
        self._dy = dy

    def __call__(self, X):
        return th.zeros(X.shape[0], self._dy, dtype=torch_dtype)


class Constant(MeanFunction):
    """
    Just a constant
    """
    def __init__(self, dy):
        super(Constant, self).__init__()
        self._dy = dy
        self._a = th.nn.Parameter(TensorType([0.0]))

    def __call__(self, x):
        return self._a.expand(x.shape[0], self._dy)


class AskeyPolynomial(MeanFunction):
    """
    Superclass for computing the tensor product of polynomials from the Askey
    scheme
    """
    def __init__(self, degree, tensor_product=False):
        """

        Args:
            degree (int): degree of polynomials
            tensor_product (bool): Whether to take the tensor product of the 1D
                basis functions.
                WARNING: for Grid inputs, the Kronecker structure of the design
                    matrix means that you HAVE to take the tensor product of the
                    basis functions associated with the various subgrids.
                    Unforutnately, this prevent an apples-to-apples comparison
                    with a normal full input (unless we define specific tensor
                    products to be taken.  Perhaps we can do this in the
                    future...)
        """
        super(AskeyPolynomial, self).__init__()
        self.f = None  # Replace with the appropriate function in subclasses
        self.degree = Param(th.IntTensor([degree]), requires_transform=False,
                            requires_grad=False)
        self.tensor_product = Param(th.ByteTensor([tensor_product]),
                                    requires_transform=False,
                                    requires_grad=False)

    def __call__(self, x):
        return self._eval(x)

    def _eval(self, x):
        """
        Evaluates the tensor product of the Hermite polynomials evaluated on X's
        columns.

        Args:
            x (Variable): columns are arguments to the Hermite polynomials
        Returns:
            (Variable)
        """
        assert(isinstance(x, Variable)), "Need Variable input"
        assert(len(x.size()) == 2), "Need a matrix as input"
        if self.tensor_product.data[0]:
            return th.cat([KroneckerProduct([
                self._eval_scalar(x[i, j]).view(1, -1)
                for j in range(x.size(1))]).eval()
                for i in range(x.size(0))])
        else:
            return th.cat([th.cat([
                self._eval_scalar(x[i, j], j == 0).view(1, -1)
                for j in range(x.size(1))], 1)
                for i in range(x.size(0))], 0)

    def _eval_scalar(self, x, include_const=True):
        """
        Evaluate polynomial terms on 1D inputs
        Args:
            x (Variable) a scalar
            include_const (bool): include the constant term or not.
        Returns:
            (Variable) A 1D tensor (length = degree + 1) of its Hermite
                polynomials.
        """
        if include_const:
            range_start = 0
        else:
            range_start = 1
        y_array = np.array([
            self.f(x.data[0], np.concatenate((np.zeros(i), [1])))
            for i in range(range_start, self.degree.data[0] + 1)])
        y = as_variable(y_array)
        return y


class HermitePolynomial(AskeyPolynomial):
    """
    Computes the tensor product of the Hermite polynomials on the columns of a
    matrix.

    This is the natural basis to use in gPC when the stochastic variables are
    Gaussian-distributed on (-inf, inf).

    Hermite polynomials:
    https://en.wikipedia.org/wiki/Hermite_polynomials#Definition

    Note: following numpy's implementation, these are the "Physicists'
    polynomials" according to Wikipedia
    """
    def __init__(self, degree, tensor_product=True):
        super(HermitePolynomial, self).__init__(degree, tensor_product)
        self.f = np.polynomial.hermite.hermval


class LegendrePolynomial(AskeyPolynomial):
    """
    Computes the tensor product of the Legendre polynomials on the columns of a
    matrix.

    This is the natural basis to use in gPC when the stochastic variables are
    uniformly-distributed on some finite interval.

    Legendre polynomials:
    https://en.wikipedia.org/wiki/Legendre_polynomials
    """
    def __init__(self, degree, tensor_product=False):
        super(LegendrePolynomial, self).__init__(degree, tensor_product)
        self.f = np.polynomial.legendre.legval


class LaguerrePolynomial(AskeyPolynomial):
    """
    Computes the tensor product of the Legendre polynomials on the columns of a
    matrix.

    This is the natural basis to use in gPC when the stochastic variables are
    exponentially-distributed on [0, inf)

    Reference:
    https://en.wikipedia.org/wiki/Laguerre_polynomials#The_first_few_polynomials
    """
    def __init__(self, degree, tensor_product=True):
        super(LegendrePolynomial, self).__init__(degree, tensor_product)
        self.f = np.polynomial.laguerre.lagval


class Additive(MeanFunction):
    def __init__(self, first_part, second_part):
        MeanFunction.__init__(self)
        self.add_1 = first_part
        self.add_2 = second_part

    def __call__(self, X):
        return th.add(self.add_1(X), self.add_2(X))


class Product(MeanFunction):
    def __init__(self, first_part, second_part):
        MeanFunction.__init__(self)

        self.prod_1 = first_part
        self.prod_2 = second_part

    def __call__(self, X):
        return th.mul(self.prod_1(X), self.prod_2(X))
