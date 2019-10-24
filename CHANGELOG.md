Current version on PyPI: 0.2.2

# 0.2.3:
* Fix gradient-shunting behavior caused by `torch.clamp()` used in `util.squared_distance()`

## Authors
* [Steven Atkinson](https://github.com/sdatkinson)

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

## Changes not breaking backward compatibility
* GPR, VFE: Allow specifying training set on .compute_loss() with x, y kwargs
* GPR, VFE: Allow specifying training inputs on ._predict() with x kwarg
