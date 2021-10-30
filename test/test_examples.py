# File: test_examples.py
# File Created: Saturday, 30th October 2021 9:04:58 am
# Author: Steven Atkinson (steven@atkinson.mn)

import os
from subprocess import check_call
from sys import executable

import pytest

args = (
    ("regression_1d.py", "--no-plot"),
    ("regression_1d.py", "--no-plot", "--model-type", "VFE"),
)


@pytest.mark.parametrize("args", args)
def test_example(args):
    basename, args = args[0], args[1:]
    script_path = os.path.join(os.path.dirname(__file__), "..", "examples", basename)
    check_call((executable, script_path) + args)


if __name__ == "__main__":
    pytest.main()
