# Version 0.3.0 Changelog
this documents tracks the changes that are new for version 0.3.0.

## Changes brekaing backward compatibility:
* GPR, VFE, SVGP: training inputs order is changed from (y, x) to (x, y) on 
    model __init__()s.

## Changes not breaking backward compatibility

* GPR, VFE: Allow specifying training set on .compute_loss() with x, y kwargs
* GPR, VFE: Allow specifying training inputs on ._predict() with x kwarg
