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
gmm.updateModel(data) // updates weights, means and variances with a single EM step given the data
```

## TODO

- optionally initialize values with k-means
- write log-likelihood method to know when to stop iterating for the optimization
- write the full optimization algorithm that terminates either with a max number of steps or stabilization of the log likelihood
