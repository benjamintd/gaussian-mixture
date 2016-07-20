'use strict'

var gaussian = require('gaussian')
var _ = require('underscore')

module.exports = GMM

function GMM (nComponents, weights, means, vars) {
  // nComponents: number of components in the mixture
  // weights: array of weights of each component in the mixture. Must sum to 1.
  // means: array of means of each component.
  // vars: array of variances of each component.
  this.nComponents = nComponents
  this.weights = weights === undefined ? _.range(nComponents).map(function () { return 1 / nComponents }) : weights
  this.means = means === undefined ? _.range(nComponents) : means
  this.vars = vars === undefined ? _.range(nComponents).map(function () { return 1 }) : vars
}

GMM.prototype.sample = function (nSamples) {
  var samples = []
  // generate gaussian models
  var gaussians = []
  for (var k = 0; k < this.nComponents; k++) {
    gaussians.push(gaussian(this.means[k], this.vars[k]))
  }
  // generate samples
  for (var i = 0; i < nSamples; i++) {
    var r = Math.random()
    var n = 0
    while (r > this.weights[n] && n < this.nComponents) {
      r -= this.weights[n]
      n++
    }
    samples.push(gaussians[n].ppf(Math.random()))
  }
  return samples
}
