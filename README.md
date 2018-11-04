# gptorch
Gaussian processes with PyTorch

## Installation

### Prerequisites

gptorch uses Python 3.6 and requires PyTorch version 0.3.1.
We're working on moving to the current stable version of PyTorch (0.4.1).

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
- Sparse GP regression (variational inducing)

## To be implemented:

- GPLVM (variational i.i.d.)
- Dynamical GP-LVM/Bayesian warped GP
- sparse GPs with SVI
- more kernels
- non-Gaussian likelihoods (e.g. for classification)
- correlated outputs
- deep GPs
