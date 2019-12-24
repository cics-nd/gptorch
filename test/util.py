"""
Utilities for unit tests
"""

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()


def needs_cuda(method):
    """
    Decorator to xfail methods that require CUDA when we don't have it.
    """
    if CUDA_AVAILABLE:
        return method
    else:
        @pytest.mark.xfail(reason="CUDA not available")
        def wrapped(obj, *args, **kwargs):
            method(obj, *args, **kwargs)
        
        return wrapped