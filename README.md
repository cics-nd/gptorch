# gptorch
[![CircleCI](https://circleci.com/gh/cics-nd/gptorch.svg?style=svg)](https://circleci.com/gh/cics-nd/gptorch)
[![codecov](https://codecov.io/gh/cics-nd/gptorch/branch/master/graph/badge.svg)](https://codecov.io/gh/cics-nd/gptorch)

Gaussian processes with PyTorch

## Installation

### Via pip

```
$ pip install gptorch
```

### From source

You can now do:
```
$ pip install -r -requirements.txt
$ python setup.py install
```

Otherwise, we've provided an `environment.yml` to make a new virtual environment with [Anaconda](https://www.anaconda.com/distribution/):

```
$ conda env create -f environment.yml
$ source activate gptorch
```

## Models implemented:

- Vanilla GP regression model
- [Sparse GP (variational free energy model)](http://www.jmlr.org/proceedings/papers/v9/titsias10a/titsias10a.pdf)
- [Sparse Variational GP](http://proceedings.mlr.press/v38/hensman15.pdf)

## To be implemented:

- [FITC sparse GPs](http://papers.nips.cc/paper/2857-sparse-gaussian-processes-using-pseudo-inputs.pdf)
- [Structured ("Kronecker") GPs](https://www.sciencedirect.com/science/article/pii/S0021999119300397)
- [Bayesian GPLVM](http://proceedings.mlr.press/v9/titsias10a/titsias10a.pdf)
- [Dynamical GP-LVM](http://papers.nips.cc/paper/4330-variational-gaussian-process-dynamical-systems)/[Bayesian warped GP](http://papers.nips.cc/paper/4494-bayesian-warped-gaussian-processes)
- Non-Gaussian likelihoods (e.g. for classification)
- Correlated outputs
- [Deep GPs](http://papers.nips.cc/paper/7045-doubly-stochastic-variational-inference-for-deep-gaussian-processes)
