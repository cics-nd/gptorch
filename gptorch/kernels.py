# File: kernels.py
# File created: Apr 13, 2017
# Author: Yinhao Zhu (yzhu10@nd.edu)

"""
Implementation of kernels for GP
"""

import torch
import numpy as np

from .util import as_tensor, squared_distance, TensorType, torch_dtype
from .model import Model
from .param import Param
from .settings import DefaultPositiveTransform


def _k_shape(X, X2):
    """
    Shape of a kernel with these inputs
    :param X:
    :param X2:
    :return:
    """
    return (X.size(0),) * 2 if X2 is None else (X.size(0), X2.size(0))


class Kernel(Model):
    """
    Base class for kernels
    """

    def __init__(self, input_dim):
        self.input_dim = int(input_dim)
        # self.name_kernel = name
        super(Kernel, self).__init__()

    def __add__(self, other):
        return Sum(self, other)

    def __mul__(self, other):
        return Product(self, other)

    def _validate_ard_shape(self, x, ARD=None):
        """
        Validates the shape of a potentially ARD hyperparameter

        :param name: The name of the parameter (used for error messages)
        :param x: A scalar or an array.
        :param ARD: None, False, or True. If None, infers ARD from value.
        :return: Tuple (value, ARD),
            val is a 1D np.ndarray
            ARD is a bool
        """
        if ARD is None:
            ARD = np.asarray(x).squeeze().shape != ()

        x = x * np.ones(self.input_dim)
        correct_shape = (self.input_dim,)

        if x.shape != correct_shape:
            raise ValueError("shape of possibly-ARD param does not match input_dim")

        return x, ARD


class Static(Kernel):
    """
    Kernels that don't depend on the value of the inputs are 'Static'.  The only
    parameter is a variance.
    """

    def __init__(self, input_dim, variance=1.0):
        super().__init__(input_dim)
        self.variance = Param(
            TensorType([variance]), transform=DefaultPositiveTransform()
        )

    def Kdiag(self, X):
        return self.variance.transform().expand(X.size(0))


class White(Static):
    """
    The White kernel
    """

    def K(self, X, X2=None, presliced=False):
        if X2 is None:
            return self.variance.transform().expand(X.size(0)).diag()
        else:
            return torch.zeros(*_k_shape(X, X2), dtype=torch_dtype)


class Constant(Static):
    """
    The Constant (aka Bias) kernel
    """

    def K(self, X, X2=None, presliced=False):
        return self.variance.transform().expand(*_k_shape(X, X2))


class Bias(Constant):
    pass


class Stationary(Kernel):
    """
    Base class for stationary kernels, which only depend on r = || x - x'||
    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(self, input_dim, variance=1.0, length_scales=None, ARD=False):
        """
        Args:
            input_dim (int): the dimension of the input
            variance (float): initial value for the signal variance
            length_scales (np.ndarray): initial value for length scale
                defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True)
            ARD (bool): ARD specifies whether the kernel has one length scale
                per dimension (ARD=True) or a single length scale (ARD=False).
        """
        super(Stationary, self).__init__(input_dim)
        self.variance = Param(
            TensorType([variance]), transform=DefaultPositiveTransform()
        )
        self.ARD = ARD
        if ARD:
            if length_scales is None:
                length_scales = np.ones(input_dim)
            elif isinstance(length_scales, np.ndarray):
                assert len(length_scales) == input_dim
            else:
                length_scales = length_scales * np.ones(input_dim)

            self.length_scales = Param(
                TensorType(length_scales), transform=DefaultPositiveTransform()
            )
        else:
            if length_scales is None:
                length_scales = 1.0
            self.length_scales = Param(
                TensorType([length_scales]), transform=DefaultPositiveTransform()
            )

    def squared_dist(self, X, X2):
        """
        Returns the SCALED squared distance between X and X2.
        """
        return (
            squared_distance(X / self.length_scales.transform())
            if X2 is None
            else squared_distance(
                X / self.length_scales.transform(), X2 / self.length_scales.transform()
            )
        )

    def dist(self, X: TensorType, X2: TensorType) -> TensorType:
        """
        Matrix of (scaled) Euclidean distances between points.

        Args:
            X: Matrix of vectors
            X2: (Optional) matrix of vectors.  If None, X2 = X.
        Returns:
            entry (i, j) is the distance between X[i, :] and X2[i, :]
        """
        # Clamp to slightly positive so that the gradient of sqrt(x) is finite
        return torch.sqrt(torch.clamp(self.squared_dist(X, X2), min=1e-40))

    def Kdiag(self, X):
        if isinstance(X, np.ndarray):
            # input is a data matrix; each row represents one instance
            X = as_tensor(X)
        # return a vector
        return self.variance.transform().expand(X.size(0))


class Exp(Stationary):
    """
    Exponential Kernel

    k(x, y; variance, length_scale) = variance * exp(-\|x-y\| / length_scale)
    """

    def K(self, X, X2=None):
        return self.variance.transform() * torch.exp(-self.dist(X, X2))


class Matern12(Exp):
    pass


class Matern32(Stationary):
    def K(self, X, X2=None):
        r = self.dist(X, X2)
        r3 = TensorType([np.sqrt(3.0)]).to(r.device) * r
        return self.variance.transform() * (1.0 + r3) * torch.exp(-r3)


class Matern52(Stationary):
    def K(self, X, X2=None):
        r = self.dist(X, X2)
        s5 = TensorType([np.sqrt(5.0)]).to(r.device)
        return (
            self.variance.transform()
            * (1.0 + s5 * r + 5.0 / 3.0 * r * r)
            * torch.exp(-s5 * r)
        )


class Rbf(Stationary):
    """
    The Radial Basis Function (RBF) or Squared Exponential / Gaussian Kernel
    """

    def K(self, X, X2=None):
        r2 = self.squared_dist(X, X2)
        return self.variance.transform() * torch.exp(-r2 / 2.0)


SquaredExponential = Rbf


class Periodic(Stationary):
    """
    Periodic kernel,
    k(r) = A cos(B * r)
    """

    def K(self, X, X2=None):
        return self.variance.transform() * torch.cos(self.dist(X, X2))


class Linear(Kernel):
    """
    The linear kernel
    """

    def __init__(self, input_dim, variance=1.0, ARD=None):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter(s)
          if ARD=True, there is one variance per input
        - active_dims is a list of length input_dim which controls
          which columns of X are used.
        """
        super().__init__(input_dim)

        variance, self.ARD = self._validate_ard_shape(variance, ARD)
        self.variance = Param(
            TensorType(variance), transform=DefaultPositiveTransform()
        )

    def K(self, X, X2=None):
        if X2 is None:
            return torch.mm(X * self.variance.transform(), X.t())
        else:
            return torch.mm(X * self.variance.transform(), X2.t())

    def Kdiag(self, X):
        return torch.sum(X * X * self.variance.transform(), 1)


class Combination(Kernel):
    """
    A combination of two kernels (e.g. Sum or Product)

    This will later get extended to arbitrary numbers of kernels, but we'll keep
    it to pairs at the moment, and you can use Combinations of Combinations to
    build more expressive kernels.
    """

    def __init__(self, kern1, kern2):
        if not kern1.input_dim == kern2.input_dim:
            raise ValueError("Kernels need the same input_dim")
        super().__init__(input_dim=kern1.input_dim)

        self.kern1 = kern1
        self.kern2 = kern2


class Product(Combination):
    """
    Product kernel

    CAREFUL: don't leave kern1 and kern2 with trainable variances unless they
    have priors!
    """

    def K(self, X, X2=None):
        return self.kern1.K(X, X2) * self.kern2.K(X, X2)

    def Kdiag(self, X):
        return self.kern1.Kdiag(X) * self.kern2.Kdiag(X)


class Sum(Combination):
    def K(self, X, X2=None):
        return self.kern1.K(X, X2) + self.kern2.K(X, X2)

    def Kdiag(self, X):
        return self.kern1.Kdiag(X) + self.kern2.Kdiag(X)
