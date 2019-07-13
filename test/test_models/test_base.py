# File: test_base.py
# File Created: Saturday, 13th July 2019 1:54:18 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest
import torch

from gptorch.models.base import GPModel
from gptorch.mean_functions import Zero
from gptorch.util import torch_dtype

torch.set_default_dtype(torch_dtype)


class TestGPModel(object):
    """
    Tests for the GPModel class
    """
    pass
