# File: test_model.py
# File Created: Saturday, 23rd February 2019 8:30:36 pm
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Tests for model.py
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch as th

from gptorch.model import Param, Model


class TestParam(object):
    """
    Tests for the Param class
    """
    def test_init(self):
        # Test various permitted inits:
        Param(th.DoubleTensor([1.0]))
        Param(th.DoubleTensor([1.0]), requires_grad=False)
        Param(th.DoubleTensor([1.0]), requires_transform=False)
        Param(th.DoubleTensor([1.0]), requires_grad=False, 
            requires_transform=False)

    def test_access(self):
        """
        Test accessing the value.
        """
        p = Param(th.DoubleTensor([1.0]))
        assert isinstance(p.data, th.DoubleTensor)
        assert isinstance(p.data.numpy(), np.ndarray)

    def test_transform(self):
        """
        Test that parameters requiring a transform return the correct value.

        Currently, we obtain the untransformed variable by default.  Perhaps we
        should switch this in the future.
        """
        p = Param(th.DoubleTensor([1.0]))
        assert p.data.numpy()[0] == 1.0
        pt = Param(th.DoubleTensor([1.0]), requires_transform=True)
        assert p.transform().data.numpy()[0] == 1.0


class TestModel(object):
    """
    Tests for the Model class
    """
    pass



