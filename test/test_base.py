
"""
Basic tests for the repo
"""


def test_torch_dtype():
    """
    Ensure that importing gptorch does not change the default dtype for torch.
    """

    import torch
    dtype = torch.get_default_dtype()
    import gptorch
    new_dtype = torch.get_default_dtype()

    assert new_dtype == dtype
