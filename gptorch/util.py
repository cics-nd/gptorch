#
# May 10, 2017  Yinhao Zhu
# Utilities: data loading, plotting, batching...
from torch.autograd import Variable
import torch
import numpy as np
from warnings import warn
from scipy.cluster.vq import kmeans2


TensorType = torch.DoubleTensor
torch_dtype = torch.double


gamma_cof=[57.1562356658629235,-59.5979603554754912,
    14.1360979747417471,-0.491913816097620199,.339946499848118887e-4,
    .465236289270485756e-4,-.983744753048795646e-4,.158088703224912494e-3,
    -.210264441724104883e-3,.217439618115212643e-3,-.164318106536763890e-3,
    .844182239838527433e-4,-.261908384015814087e-4,.368991826595316234e-5]


def tensor_type():
    warn("Use TensorType instead of tensor_type()", DeprecationWarning)
    return TensorType


def as_variable(x):
    """
    Convert a numpy matrix to a Tensor variable

    Args:
        x (np.ndarray)
    Returns:
        (TensorType)
    """
    warn("Deprecated.  Use as_tensor() instead.", DeprecationWarning)
    return as_tensor(x)


def as_tensor(x):
    """
    Convert a numpy matrix into a raw Tensor.
    If you're unsure, use as_variable() instead.

    Args:
        x (np.ndarray)
    Returns:
        (TensorType)
    """
    if isinstance(x, torch.Tensor):
        return x.type(TensorType)
    if isinstance(x, np.ndarray):
        return TensorType(x)
    elif isinstance(x, float):
        return TensorType([x])
    else:
        raise TypeError("Unsupported type {}".format(type(x)))


def KL_Gaussian(m1, m2, S1, S2):
    """
    KL-divergence between two Gaussians
    $p\sim\mathcal{N}(m1, S1), q\sim\mathcal{N}(m2, S2)$
    $\textrm{KL}(p||q) = \frac{1}{2}(\log\frac{|\Sigma_2|}{\Sigma_1} - n + \text{Tr}((\Sigma_2^{-1}\Sigma_1) + (m_2-m_1)^T\Sigma_2^{-1}(m_2-m_1)$
    :param m1:
    :param m2:
    :param S1:
    :param S2:
    :return:
    """
    raise NotImplementedError("")


def kmeans_centers(x: np.ndarray, k: int) -> np.ndarray:
    """
    Use k-means clustering and find the centers of the clusters.
    :param x: The data
    :param k: Number of clusters
    :return: the centers
    """
    return kmeans2(x, k)[0]


def PCA(X, q):
    """
    A helpful function for linearly reducing the dimensionality of the data X
    to q.

    Args:
        X (np.ndarray(n, p)): Observations
        q (int): Dimensionality of latent variable

    Returns:
        (np.ndarray(n, q)): PCA projection array of Z

    """
    assert q <= X.shape[1], 'Cannot have more latent dimensions than observed'
    evals, evecs = np.linalg.eigh(np.cov(X.T))
    indices = np.argsort(evals)[::-1]
    W = evecs[:, indices]
    W = W[:, :q]
    return (X - X.mean(0)).dot(W)


def gammaln(xx):
    """
    Used to approximate the gamma function using the Lanczos algorithm
    Ref: Numerical Methods (Third Edition) by William Press et al. pg. 256

    Args:
        xx (Variable): desired gamma value, must be > 0
    Returns:
        ln(gamma(xx)) (Variable): natural log of desired gamma value
    """

    if (xx <= 0):
        raise ValueError('Illegal value for gammaln!')
    y = x = xx
    tmp = x + 5.24218750000000000    #Rational 671/128.
    tmp = (x+0.5)*np.log(tmp)-tmp
    ser = 0.999999999999997092 #First coefficiect
    for i in range(0,14):
        ser += gamma_cof[i]/(y + i + 1)
    return tmp+np.log(2.5066282746310005*ser/x)


def squared_distance(x1: TensorType, x2: TensorType=None) -> TensorType:
    """
    Given points x1 [n1 x d1] and x2 [n2 x d2], return a [n1 x n2] matrix with
    the pairwise squared distances between the points.

    Entry (i, j) is sum_{j=1}^d (x_1[i, j] - x_2[i, j]) ^ 2
    """
    if x2 is None:
        return squared_distance(x1, x1)
    x1s = x1.pow(2).sum(1, keepdim=True)
    x2s = x2.pow(2).sum(1, keepdim=True)
    r2 = x1s + x2s.t() -2.0 * x1 @ x2.t()
    # Prevent negative squared distances using torch.clamp
    # NOTE: Clamping is for numerics.
    # This use of .detach() is to avoid breaking the gradient flow.
    return r2 - (torch.clamp(r2, max=0.0)).detach()
