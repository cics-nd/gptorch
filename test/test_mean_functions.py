# File: test_mean_functions.py
# File Created: Saturday, 13th July 2019 3:51:40 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest
import torch

from gptorch import mean_functions


class TestConstant(object):
    def test_init(self):
        dy = 2
        mean_functions.Constant(dy)
        mean_functions.Constant(dy, val=torch.randn(dy))

        with pytest.raises(ValueError):
            # dim doesn't match val
            mean_functions.Constant(dy, val=torch.Tensor([1.0]))

    def test_forward(self):
        n, dx, dy = 5, 3, 2

        y = mean_functions.Constant(dy)(torch.rand(n, dx))
        assert isinstance(y, torch.Tensor)
        assert all([e == a for e, a in zip(y.flatten(), torch.zeros(n, dy).flatten())])

        val = torch.randn(dy)
        y = mean_functions.Constant(dy, val=val)(torch.rand(n, dx))
        assert isinstance(y, torch.Tensor)
        assert all(
            [e == a for e, a in zip(y.flatten(), (val + torch.zeros(n, dy)).flatten())]
        )


class TestZero(object):
    def test_init(self):
        mean_functions.Zero(2)

    def test_forward(self):
        n, dx, dy = 5, 3, 2
        y = mean_functions.Zero(dy)(torch.rand(n, dx))
        assert isinstance(y, torch.Tensor)
        assert all([e == a for e, a in zip(y.flatten(), torch.zeros(n, dy).flatten())])
