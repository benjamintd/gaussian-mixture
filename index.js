'use strict'

var gaussian = require('gaussian')
var _ = require('underscore')

module.exports = GMM

function GMM (nComponents, weights, means, vars) {
  // nComponents: number of components in the mixture
  // weights: array of weights of each component in the mixture, must sum to 1
  // means: array of means of each component
  // vars: array of variances of each component
  this.nComponents = nComponents
  this.weights = weights === undefined ? _.range(nComponents).map(function () { return 1 / nComponents }) : weights
  this.means = means === undefined ? _.range(nComponents) : means
  this.vars = vars === undefined ? _.range(nComponents).map(function () { return 1 }) : vars
  if (nComponents !== this.weights.length ||
    nComponents !== this.means.length ||
    nComponents !== this.vars.length) {
    throw new Error('weights, means and vars must have nComponents elements.')
  }
}

GMM.prototype.sample = function (nSamples) {
  // nSamples: integer
  var samples = []
  var gaussians = this._gaussians()
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

GMM.prototype.scoreSamples = function (data) {
  // data: array of floats representing the samples to score under the model
}

GMM.prototype.scoreSample = function (x) {
  // x: a float representing a single datapoint
  var probs = []
  var gaussians = this._gaussians()
  for (var i = 0; i < this.nComponents; i++) {
    probs.push(gaussians[i].pdf(x))
  }
  var sum = probs.reduce(function (a, b) { return a + b })
  return probs.map(function (a) {
    return a / sum
  })
}

GMM.prototype._gaussians = function () {
  var gaussians = []
  for (var k = 0; k < this.nComponents; k++) {
    gaussians.push(gaussian(this.means[k], this.vars[k]))
  }
  return gaussians
}
