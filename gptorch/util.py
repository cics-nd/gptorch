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


def as_tensor(x):
    """
    Convert a numpy matrix into a Tensor.

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


def kmeans_centers(x: np.ndarray, k: int, perturb_if_fail: bool=False) -> \
        np.ndarray:
    """
    Use k-means clustering and find the centers of the clusters.
    :param x: The data
    :param k: Number of clusters
    :param perturb_if_fail: Move the points randomly in case of a numpy 
        LinAlgError.
    :return: the centers
    """
    try:
        return kmeans2(x, k)[0]
    except np.linalg.LinAlgError:
        x_scale = x.std(axis=0)
        x_perturbed = x + 1.0e-4 * x_scale * np.random.randn(*x.shape)
        return kmeans2(x_perturbed, k)[0]


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
    assert q <= X.shape[1], "Cannot have more latent dimensions than observed"
    evals, evecs = np.linalg.eigh(np.cov(X.T))
    indices = np.argsort(evals)[::-1]
    W = evecs[:, indices]
    W = W[:, :q]
    return (X - X.mean(0)).dot(W)


def squared_distance(x1: TensorType, x2: TensorType = None) -> TensorType:
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
