# File: param.py
# File Created: Wednesday, 10th July 2019 11:41:35 pm
# Author: Steven Atkinson (steven@atkinson.mn)

"""
param.py: Parameters
"""

import torch
from torch.distributions.transforms import ComposeTransform


class Param(torch.nn.Parameter):
    """
    Customized Parameter class extending the PyTorch Parameter class.
    Its main purpose is to include the following additional functionality:
    1) The .transform() member function, in order to impose constraints on the 
        parameter.  Use torch.distributions.transforms classes for this.
    2) the .prior member, for incorporation into joint log-probabilities (e.g. 
        for training)
    """
    def __new__(cls, data=None, requires_grad=True, transform=None,
                prior=None):
        transform = ComposeTransform([]) if transform is None else transform
        data = transform.inv(data)
        return super().__new__(cls, data, requires_grad=requires_grad)

    def __init__(self, data, requires_grad=True, transform=None):
        super().__init__()
        transform = ComposeTransform([]) if transform is None else transform
        self._transform = transform
        self.prior = None

    def transform(self):
        return self._transform(self)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()
