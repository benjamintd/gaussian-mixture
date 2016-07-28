# Gaussian Mixture

This module implements a 1D Gaussian Mixture class that allows to fit a distribution of points along a one dimensional axis.

## Usage

```
var GMM = require('gaussianMixture')

var nComponents = 3
var weights = [0.3, 0.2, 0.5]
var means = [1, 7, 15]
var vars = [1, 2, 1]

var gmm = new GMM(nComponents, weights, means, vars)
gmm.sample(10)  // return 10 datapoints from the mixture

data = [1.2, 1.3, 7.4, 1.4, 14.3, 15.3, 1.0, 7.2]
gmm.optimize(data) // updates weights, means and variances with the EM algorithm given the data.
```
