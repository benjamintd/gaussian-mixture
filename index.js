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
  // returns: array of length nSamples
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

GMM.prototype.memberships = function (data) {
  // data: array of floats representing the samples to score under the model
  // return: (data.length * this.nComponents) matrix with membership weights
  var memberships = []
  for (var i = 0; i < data.length; i++) {
    memberships.push(this.membership(data[i]))
  }
  return memberships
}

GMM.prototype.membership = function (x) {
  // x: float representing a single datapoint
  // return: array of probabilities of length this.nComponents
  var membership = []
  var gaussians = this._gaussians()
  for (var i = 0; i < this.nComponents; i++) {
    membership.push(gaussians[i].pdf(x))
  }
  var sum = membership.reduce(function (a, b) { return a + b })
  return membership.map(function (a) { return a / sum })
}

GMM.prototype.updateModel = function (data) {
  // This function performs an expectation maximization step.
  // It will update the GMM weights, means and variances.

  // First, we compute the data memberships.
  var n = data.length
  var memberships = this.memberships(data)

  // Update the mixture weights
  var componentWeights = []
  for (var k = 0; k < this.nComponents; k++) {
    componentWeights[k] = memberships.reduce(function (a, b) { return (a + b[k]) }, 0)
  }
  this.weights = componentWeights.map(function (a) { return a / n })

  // Update the mixture means
  for (k = 0; k < this.nComponents; k++) {
    this.means[k] = 0
    for (var i = 0; i < n; i++) {
      this.means[k] += memberships[i][k] * data[i]
    }
    this.means[k] /= componentWeights[k]
  }

  // Update the mixture variances
  for (k = 0; k < this.nComponents; k++) {
    this.vars[k] = 0
    for (i = 0; i < n; i++) {
      this.vars[k] += memberships[i][k] * Math.pow(data[i] - this.means[k], 2)
    }
    this.vars[k] /= componentWeights[k]
  }
}

GMM.prototype._gaussians = function () {
  var gaussians = []
  for (var k = 0; k < this.nComponents; k++) {
    gaussians.push(gaussian(this.means[k], this.vars[k]))
  }
  return gaussians
}
