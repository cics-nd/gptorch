#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import sys

valid_python = sys.version_info[0] >= 3 and sys.version_info[1] >= 5
if not valid_python:
    raise RuntimeError("gptorch requires python 3.5+")

# Python 3.5 can't do matplotlib 3.1 and up.
matplotlib_dependency = "matplotlib>=2.1.2" if sys.version_info[1] >= 6 else \
    "matplotlib>=2.1.2, <3.1"

requirements = [
    "numpy>=1.10",
    "scipy>=0.18",
    matplotlib_dependency,
    "scikit-learn>=0.19.1",
    "torch",  # conda install pytorch -c pytorch
    "pytest>=3.5.0",
    "graphviz>=0.9",
    "jupyter"
]

setup(name="gptorch",
    version="0.3.0",
    description="gptorch - a Gaussian process toolbox built on PyTorch",
    author="Yinhao Zhu, Steven Atkinson",
    author_email="yzhu10@nd.edu, satkinso@nd.edu",
    url="https://github.com/cics-nd/gptorch",
    install_requires=requirements,
    packages=find_packages()
)
