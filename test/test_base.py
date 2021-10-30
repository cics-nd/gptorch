# File: test_base.py
# File Created: Saturday, 30th October 2021 7:39:23 am
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Basic tests for the repo
"""


def test_torch_dtype():
    """
    Ensure that importing gptorch does not change the default dtype for torch.
    """

    import torch

    dtype = torch.get_default_dtype()
    import gptorch  # noqa F401

    new_dtype = torch.get_default_dtype()

    assert new_dtype == dtype
