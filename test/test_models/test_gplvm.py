# File: test_gplvm.py
# File Created: Saturday, 19th January 2020 9:20:18 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import os
import sys

import pytest
import torch
import numpy as np

from gptorch.models.base import GPModel
from gptorch.mean_functions import Zero
from gptorch.util import TensorType
from gptorch.kernels import Rbf
from gptorch.models.gplvm import GPLVM
from gptorch.util import torch_dtype


class TestGPLVM(object):
    def test_init(self):
        def _trial(n, dz, dx):
            torch.manual_seed(42)

            m = n // 2
            large_p = dx > n

            z = torch.randn(n, dz, dtype=torch_dtype)
            w = torch.randn(dz, dx, dtype=torch_dtype)
            b = torch.randn(1, dx, dtype=torch_dtype)
            x = z @ w + b

            model = GPLVM(x.numpy(), dz, m, large_p=large_p)

        _trial(5, 2, 3)
        _trial(5, 2, 7)
