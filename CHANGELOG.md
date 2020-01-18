Current version on PyPI: 0.3.1

# 0.4.0:

* `gptorch.functions.cholesky` and `inverse` work on batches of matrices.
* Handle main functionality in `likelihoods.Likelihood` with `.forward()`, 
  `.log_marginal_likelihood()`, and `.marginal_log_likelihood()`.
* Add likelihoods: `Bernoulli`, `Binomial`
* `GPModel.loss()` returns a scalar, not rank-1 Tensor.
