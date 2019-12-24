Current version on PyPI: 0.2.3

# 0.3.0:

## Changes breaking backward compatibility:
* GPR, VFE, SVGP: training inputs order is changed from (y, x) to (x, y) on 
    model __init__()s.
* `.predict()` functions return the same type as the inputs provided 
    (numpy.ndarray->numpy.ndarray, torch.Tensor->torch.Tensor)
* Remove `util.as_variable()`
* Remove `util.tensor_type()`
* Remove `util.KL_Gaussian()`
* Remove `util.gammaln()`
* GPModel method `.loss()` generally replaces `.compute_loss()`.
* `.compute_loss()` methods in models generally renamed to `.log_likelihood()` 
  and signs flipped to reflect the fact that the loss is generally the negative
  LL.

## Changes not breaking backward compatibility
* GPR, VFE: Allow specifying training set on .compute_loss() with x, y kwargs
* GPR, VFE: Allow specifying training inputs on ._predict() with x kwarg
* GPU supported with .cuda()
* Eliminate GPModel.evaluate()
* Don't print inducing inputs on sparse GP initialization
* Suport for priors in `gptorch.model.Model`s
