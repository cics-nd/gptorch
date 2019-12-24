# File: __init__.py
# Author: Yinhao Zhu (yzhu10@nd.edu)

from .base import GPModel

"""
Core Gaussian process models.

Models typically comprise of a GP-based (latent) function combined with some 
likelihood whose density can be evaluated at target (output) locations.

GPR implements (exact) Guassian process regression

sparse_gpr contains the variational free energy (VFE) model of Titsias (2009)
and the sparse variational GP (SVGP) model based on Hensman et al. (2015).
The latter is special in that it is tractably compatible with non-conjugate 
likelihoods, making it usable for non-regression tasks (e.g. classification).
"""

from .gpr import GPR
from .sparse_gpr import VFE, SVGP
