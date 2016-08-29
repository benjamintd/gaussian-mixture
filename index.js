'use strict';

// Imports
var gaussian = require('gaussian');
var barycenter = require('./utilities/barycenter');
var _ = require('underscore');

// Constants
var MAX_ITERATIONS = 200;
var LOG_LIKELIHOOD_TOL = 1e-7;

module.exports = GMM;

function GMM(nComponents, weights, means, vars, options) {
  // nComponents: number of components in the mixture
  // weights: array of weights of each component in the mixture, must sum to 1
  // means: array of means of each component
  // vars: array of variances of each component
  this.nComponents = nComponents;
  this.weights = weights === undefined ? _.range(nComponents).map(function () { return 1 / nComponents; }) : weights;
  this.means = means === undefined ? _.range(nComponents) : means;
  this.vars = vars === undefined ? _.range(nComponents).map(function () { return 1; }) : vars;
  if (nComponents !== this.weights.length ||
      nComponents !== this.means.length ||
      nComponents !== this.vars.length) {
    throw new Error('weights, means and vars must have nComponents elements.');
  }
  this.options = options === undefined ? {} : options;
}

GMM.prototype._gaussians = function () {
  var gaussians = [];
  for (var k = 0; k < this.nComponents; k++) {
    gaussians.push(gaussian(this.means[k], this.vars[k]));
  }
  return gaussians;
};

GMM.prototype.sample = function (nSamples) {
  // nSamples: integer
  // return: array of length nSamples
  var samples = [];
  var gaussians = this._gaussians();
  // generate samples
  for (var i = 0; i < nSamples; i++) {
    var r = Math.random();
    var n = 0;
    while (r > this.weights[n] && n < this.nComponents) {
      r -= this.weights[n];
      n++;
    }
    samples.push(gaussians[n].ppf(Math.random()));
  }
  return samples;
};

GMM.prototype.memberships = function (data) {
  // data: array of floats representing the samples to score under the model
  // return: (data.length * this.nComponents) matrix with membership weights
  var memberships = [];
  for (var i = 0, n = data.length; i < n; i++) {
    memberships.push(this.membership(data[i]));
  }
  return memberships;
};

GMM.prototype.membership = function (x) {
  // x: float representing a single datapoint
  // return: array of probabilities of length this.nComponents
  var membership = [];
  var gaussians = this._gaussians();
  for (var i = 0; i < this.nComponents; i++) {
    membership.push(gaussians[i].pdf(x));
  }
  var sum = membership.reduce(function (a, b) { return a + b; });
  return membership.map(function (a) { return a / sum; });
};

GMM.prototype.updateModel = function (data) {
  // This function performs an expectation maximization step.
  // It will update the GMM weights, means and variances.
  // Optionally, if options.variancePrior and options.priorRelevance are defined,
  // mix in the prior.

  // First, we compute the data memberships.
  var n = data.length;
  var memberships = this.memberships(data);
  var alpha;

  // Update the mixture weights
  var componentWeights = [];
  var reduceFunction = function (k) { return function (a, b) { return (a + b[k]); }; };
  for (var k = 0; k < this.nComponents; k++) {
    componentWeights[k] = memberships.reduce(reduceFunction(k), 0);
  }
  this.weights = componentWeights.map(function (a) { return a / n; });

  // Update the mixture means
  for (k = 0; k < this.nComponents; k++) {
    this.means[k] = 0;
    for (var i = 0; i < n; i++) {
      this.means[k] += memberships[i][k] * data[i];
    }
    this.means[k] /= componentWeights[k];
  }
  // If there is a separation prior:
  if (this.options.separationPrior && this.options.priorRelevance) {
    var separationPrior = this.options.separationPrior;
    var priorMeans = _.range(this.nComponents).map(function (a) { return (a * separationPrior); });
    var priorCenter = barycenter(priorMeans, this.weights);
    var center = barycenter(this.means, this.weights);
    for (k = 0; k < this.nComponents; k++) {
      alpha = this.weights[k] / (this.weights[k] + this.options.priorRelevance);
      this.means[k] = center + alpha * (this.means[k] - center) + (1 - alpha) * (priorMeans[k] - priorCenter);
    }
  }

  // Update the mixture variances
  for (k = 0; k < this.nComponents; k++) {
    this.vars[k] = 0;
    for (i = 0; i < n; i++) {
      this.vars[k] += memberships[i][k] * Math.pow(data[i] - this.means[k], 2);
    }
    this.vars[k] /= componentWeights[k];
    // If there is a variance prior:
    if (this.options.variancePrior && this.options.priorRelevance) {
      alpha = this.weights[k] / (this.weights[k] + this.options.priorRelevance);
      this.vars[k] = alpha * this.vars[k] + (1 - alpha) * this.options.variancePrior;
    }
  }
};

GMM.prototype.logLikelihood = function (data) {
  // data: array of floats representing the samples to compute the log-likelihood with.
  var l = 0;
  var p = 0;
  var gaussians = this._gaussians();
  for (var i = 0, n = data.length; i < n; i++) {
    p = 0;
    for (var k = 0; k < this.nComponents; k++) {
      p += this.weights[k] * gaussians[k].pdf(data[i]);
    }
    l += Math.log(p);
  }
  return l;
};

GMM.prototype.optimize = function (data, maxIterations, logLikelihoodTol) {
  // data: array of floats
  maxIterations = maxIterations === undefined ? MAX_ITERATIONS : maxIterations;
  logLikelihoodTol = logLikelihoodTol === undefined ? LOG_LIKELIHOOD_TOL : logLikelihoodTol;
  var logLikelihoodDiff = Infinity;
  var logLikelihood = -Infinity;
  var temp;
  for (var i = 0; i < maxIterations && logLikelihoodDiff > logLikelihoodTol; i++) {
    this.updateModel(data);
    temp = this.logLikelihood(data);
    logLikelihoodDiff = Math.abs(logLikelihood - temp);
    logLikelihood = temp;
  }
  return i; // number of steps to reach the converged solution.
};
