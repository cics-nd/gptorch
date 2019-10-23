# File: test_util.py
# File Created: Wednesday, 23rd October 2019 2:54:13 pm
# Author: Steven Atkinson (212726320@ge.com)

import os
import sys

import pytest
import torch

base_path = os.path.join(os.path.dirname(__file__), "..")
if not base_path in sys.path:
    sys.path.append(base_path)

from gptorch import util


class TestSquaredDistance(object):
    """
    util.squared_distance

    There's a lot riding on getting this right!
    """

    def test_type(self):
        x1, x2 = self._vals_1d()

        r2_actual = util.squared_distance(x1, x2)
        assert isinstance(r2_actual, util.TensorType)

    def test_shape(self):
        x1, x2 = self._vals_1d()

        r2_actual = util.squared_distance(x1, x2)
        assert r2_actual.shape[0] == x1.shape[0]
        assert r2_actual.shape[1] == x2.shape[0]

    def test_values(self):
        x1, x2 = self._vals_1d()

        r2_actual = util.squared_distance(x1, x2)
        r2_expected = util.TensorType([[0.0, 4.0, 16.0], [1.0, 1.0, 9.0], [4.0, 0.0, 4.0]])
        assert all([a.item() == pytest.approx(e.item()) 
            for a, e in zip(r2_actual.flatten(), r2_expected.flatten())])

    def test_grads_1_nonzero(self):
        """
        Ensure correct first derivatives.
        No major challenges here...
        Case where it's nonzero (point1 and point2 are distinct)
        """
        x1, x2 = self._vals_1d()
        x1.requires_grad_(True)

        r2_actual = util.squared_distance(x1, x2)

        # val = (0-2)^2, grad=2(0-2)=-4
        r2_actual[0, 1].backward(retain_graph=True)  
        grad01_actual = x1.grad[0].item()
        grad01_expected = -4.0
        assert grad01_actual == grad01_expected

    def test_grads_1_zero(self):
        """
        Ensure correct first derivatives.
        No major challenges here...
        Case where it's zero (point1 = point2)
        """
        x1, x2 = self._vals_1d()
        x1.requires_grad_(True)

        r2_actual = util.squared_distance(x1, x2)

        # val = (0-0)^2, grad=2(0-0)=0
        r2_actual[0, 0].backward(retain_graph=True)
        grad00_actual = x1.grad[0].item()
        grad00_expected = 0.0
        assert grad00_actual == grad00_expected

    def test_grads_2(self):
        """
        Ensure correct second derivatives.
        This is nontrivial because the gradient at squared distance~0 could get 
        erased by a .clamp() if you code it up wrong!
        """

        x1, x2 = self._vals_1d()
        x1.requires_grad_(True)
         # Trick to track grads to single rows:
        x1_rows = [xi for xi in x1]
        x1 = torch.stack(x1_rows)

        r2_actual = util.squared_distance(x1, x2)
        r2_00 = r2_actual[0, 0]
        # First derivative:
        # Returns a tuple w/ one entry that's a tensor of size (1,)
        drdx = torch.autograd.grad(r2_00, x1_rows[0], create_graph=True)
        # Second derivative:
        d2rdx2_actual = torch.autograd.grad(drdx[0][0], x1_rows[0])

        # val = (x1-x2)^2
        # 1st der = 2*(x1-x2) = 2(0-0) = 0
        # 2nd der = 2  <-- but this will show up as zero instead if it got clamped!
        d2rdx2_expected = 2.0

        assert d2rdx2_actual[0].item() == d2rdx2_expected

    @staticmethod
    def _vals_1d():
        x1 = util.TensorType([[0.0], [1.0], [2.0]]) + 1.0 / 65.0
        x2 = util.TensorType([[0.0], [2.0], [4.0]]) + 1.0 / 65.0

        return x1, x2
