# File: test_gpr.py
# File Created: Saturday, 13th July 2019 3:25:43 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest
import numpy as np
import torch

from gptorch.models import GPR
from gptorch.kernels import Rbf


class TestGPR(object):
    def test_init(self):
        n, dx, dy = 5, 3, 2
        x, y = np.random.randn(n, dx), np.random.randn(n, dy)
        kern = Rbf(x.shape[1], ARD=True)

        # init w/ numpy
        GPR(y, x, kern)
        # init w/ PyTorch tensors:
        GPR(torch.Tensor(y), torch.Tensor(x), kern)
        # init w/ a mean function:
        GPR(y, x, kern, mean_function=torch.nn.Linear(dx, dy))

    @pytest.mark.xfail()
    def test_compute_loss(self):
        raise NotImplementedError()

    @pytest.mark.xfail()
    def test_predict(self):
        raise NotImplementedError()
