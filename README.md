# gptorch
Gaussian processes with PyTorch

## Installation

The simplest way to install is via [Anaconda](https://www.anaconda.com/distribution/) using the provided `environment.yml`:

```
$ conda env create -f environment.yml
$ source activate gptorch
```

## Models implemented:

- GP regression
- Sparse GP regression (variational inducing points)

## To be implemented:

- [FITC sparse GPs](http://papers.nips.cc/paper/2857-sparse-gaussian-processes-using-pseudo-inputs.pdf)
- [Sparse GPs with SVI](http://proceedings.mlr.press/v38/hensman15.pdf)
- [Structured ("Kronecker") GPs](https://www.sciencedirect.com/science/article/pii/S0021999119300397)
- [Bayesian GPLVM](http://proceedings.mlr.press/v9/titsias10a/titsias10a.pdf)
- [Dynamical GP-LVM](http://papers.nips.cc/paper/4330-variational-gaussian-process-dynamical-systems)/[Bayesian warped GP](http://papers.nips.cc/paper/4494-bayesian-warped-gaussian-processes)
- Non-Gaussian likelihoods (e.g. for classification)
- Correlated outputs
- [Deep GPs](http://papers.nips.cc/paper/7045-doubly-stochastic-variational-inference-for-deep-gaussian-processes)
