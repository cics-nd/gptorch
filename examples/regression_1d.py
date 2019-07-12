# Steven Atkinson
# satkinso@nd.edu
# July 19, 2018

"""
Demonstration of GPs for regression
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import matplotlib.pyplot as plt
import numpy as np

from gptorch.models.gpr import GPR
from gptorch.models.sparse_gpr import VFE
from gptorch import kernels
from gptorch.util import TensorType
from gptorch import mean_functions

torch.manual_seed(42)
np.random.seed(42)

# Data
def f(x):
    return np.sin(2. * np.pi * x) + np.cos(3.5 * np.pi * x) - 3.0 * x + 5.0


if __name__ == "__main__":
    # Create data:
    n = 100
    x = np.linspace(0, 1, n).reshape((-1, 1))
    y = f(x) + 0.1 * np.random.randn(n, 1)

    # Try different kernels...
    # kern = kernels.Rbf(1)
    # kern = kernels.Matern32(1)
    # kern = kernels.Matern52(1)
    # kern = kernels.Exp(1)
    # kern = kernels.Constant(1)
    # kern = kernels.Linear(1)
    kern = kernels.Linear(1) + kernels.Rbf(1) + kernels.Constant(1)

    # Try different models:
    model = GPR(y, x, kern)
    # model = VFE(y, x, kern)
    model.likelihood.variance.data = TensorType([1.0e-6])

    # Train
    model.optimize(method="L-BFGS-B", max_iter=100)
    print("Trained model:")
    print(model)

    # Predict
    n_test = 200
    n_samples = 5
    x_test = np.linspace(-1, 2, n_test).reshape((-1, 1))
    mu, s = model.predict_y(x_test)
    mu, s = mu.detach().numpy().flatten(), s.detach().numpy().flatten()
    y_samp = model.predict_y_samples(x_test, n_samples).detach().numpy()
    unc = 2.0 * np.sqrt(s)

    # Show prediction
    x_test = x_test.flatten()
    plt.figure()
    plt.fill_between(x_test, mu - unc, mu + unc, color=(0.9,) * 3)
    plt.plot(x_test, mu)
    plt.plot(x_test, f(x_test))
    for y_samp_i in y_samp:
        plt.plot(x_test, y_samp_i, color=(0.4, 0.7, 1.0), alpha=0.5)
    plt.plot(x, y, 'o')
    plt.show()
