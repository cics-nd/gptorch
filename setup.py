#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

requirements = [
    "numpy>=1.10",
    "scipy>=0.18",
    "matplotlib>=2.1.2",
    "scikit-learn>=0.19.1",
    # "pytorch==0.3.1",  # conda install pytorch=0.3.1 -c pytorch
    "pytest>=3.5.0",
    "graphviz>=0.9"
]

# PyTorch:
# conda install pytorch=0.3.1 -c pytorch

setup(name="gptorch",
    version="0.1.0",
    description="GPtorch - a Gaussian process toolbox built on PyTorch",
    author="Yinhao Zhu, Steven Atkinson",
    author_email="yzhu10@nd.edu, satkinso@nd.edu",
    url="https://github.com/cics-nd/gptorch",
    install_requires=requirements,
    packages=find_packages(),
    )
