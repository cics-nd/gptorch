# File: test_util.py
# File Created: Sunday, 19th January 2020 4:17:01 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import os
import sys

import numpy as np
import pytest
import torch

base_path = os.path.join(os.path.dirname(__file__), "..")
if not base_path in sys.path:
    sys.path.append(base_path)

from gptorch import functions


class TestJitOp(object):
    def test_call(self):
        # Just works
        functions.jit_op(lambda x: x, torch.eye(3))

        # Verbose
        functions.jit_op(lambda x: x, torch.eye(3), verbose=True)

        # Works with batches of matrices
        functions.jit_op(lambda x: x, torch.ones(5, 3, 3))

        with pytest.raises(RuntimeError):
            functions.jit_op(lambda x: torch.cholesky(x), np.nan * torch.ones(3))

    def test_call_private(self):
        functions._jit_op(lambda x: x, torch.eye(3), 3, False)
        functions._jit_op(lambda x: x, torch.eye(3), 3, True)


def test_cholesky():
    # Works:
    functions.cholesky(torch.eye(3))
    # Works if we need to add jitter
    functions.cholesky(torch.ones(3, 3))
