# gptorch
Gaussian processes with PyTorch

## Installation

The simplest way to get set up is to use the provided `environment.yml`, i.e.,

```
conda env create -f enviroment.yml
```

Otherwise, read on...

### Prerequisites

gptorch uses Python 3.6 and currently requires PyTorch version 0.3.1.
We're working on moving to the current stable version of PyTorch.

To install PyTorch for CPU only, use

```
conda install pytorch=0.3.1 -c pytorch
```

### 2. Install gptorch

Install with

```
python setup.py install
```

## Models implemented:

- GP regression
- Sparse GP regression (variational inducing points)

## To be implemented:

- GPLVM (variational i.i.d.)
- Dynamical GP-LVM/Bayesian warped GP
- sparse GPs with SVI
- more kernels
- non-Gaussian likelihoods (e.g. for classification)
- correlated outputs
- deep GPs
