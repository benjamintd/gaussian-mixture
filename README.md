# Gaussian Mixture
[![Build Status](https://travis-ci.org/benjamintd/gaussian-mixture.svg?branch=master)](https://travis-ci.org/benjamintd/gaussian-mixture)

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
console.log(gmm.means); // >> [1.225, 7.3, 14.8]
```

## Options

Optionally, define an options object and pass it at instantiation:

```
var options = {
  variancePrior: // Float,
  separationPrior: // Float,
  priorRelevance: // Positive float.
};

var gmm = new GMM(nComponents, weights, means, vars, options);
```

The variance prior allows you to define a prior for the variance of all components (assumed the same). This prior will be mixed in the maximization step of the EM optimization with a relevance score equal to `options.priorRelevance`.

Similarly, the separation prior allows you to define a prior for the difference between consecutive gaussian mixture means. This prior is mixed with the sale `options.priorRelevance` score.

The mixing weight `alpha` for component `i` is `alpha = weights[i] / (weights[i] + options.priorRelevance)`.
