'use strict';

// Imports
var gaussian = require('gaussian');
var barycenter = require('./utilities/barycenter');
var _ = require('underscore');

// Constants
var MAX_ITERATIONS = 200;
var LOG_LIKELIHOOD_TOL = 1e-7;

module.exports = GMM;

/**
 * Instantiate a new GMM.
 * @param {Number} nComponents number of components in the mixture
 * @param {Array} weights array of weights for each component in the mixture, must sum to 1
 * @param {Array} means array of means for each component
 * @param {Array} vars array of variances of each component
 * @param {Object} options an object that can define the `variancePrior`, `separationPrior`, `variancePriorRelevance` and `separationPriorRelevance`.
 * The priors are taken into account when the GMM is optimized given some data. The relevance parameters should be non-negative numbers,
 * 1 meaning that the prior has equal weight as the result of the optimal GMM in each EM step, 0 meaning no influence, and Infinity means a fixed variance (resp. separation).
 * @example var gmm = new GMM(3, [0.3, 0.2, 0.5], [1, 2, 3], [1, 1, 0.5]);
 */
function GMM(nComponents, weights, means, vars, options) {
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

/**
 * @private
 * Return an array of gaussians for the given GMM.
 * @return {Array} an arrau of gaussian objects. For more information on how to use those gaussians, see https://www.npmjs.com/package/gaussian
 */
GMM.prototype._gaussians = function () {
  var gaussians = [];
  for (var k = 0; k < this.nComponents; k++) {
    gaussians.push(gaussian(this.means[k], this.vars[k]));
  }
  return gaussians;
};

/**
 * Randomly sample from the GMM's distribution.
 * @param {Number} nSamples desired number of samples
 * @return {Array} An array of randomly sampled numbers that follow the GMM's distribution
 */
GMM.prototype.sample = function (nSamples) {
  var samples = [];
  var gaussians = this._gaussians();

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

/**
 * Given an array of data, determine their memberships for each component of the GMM.
 * @param {Array} data array of numbers representing the samples to score under the model
 * @return {Array} (data.length * this.nComponents) matrix with membership weights
 */
GMM.prototype.memberships = function (data) {
  var memberships = [];
  for (var i = 0, n = data.length; i < n; i++) {
    memberships.push(this.membership(data[i]));
  }
  return memberships;
};

/**
 * Given a datapoint, determine its memberships for each component of the GMM.
 * @param {Number} x number representing the sample to score under the model
 * @return {Array} an array of length this.nComponents with membership weights, i.e the probabilities that this datapoint was drawn from the each component
 */
GMM.prototype.membership = function (x) {
  var membership = [];
  var gaussians = this._gaussians();
  for (var i = 0; i < this.nComponents; i++) {
    membership.push(gaussians[i].pdf(x));
  }
  var sum = membership.reduce(function (a, b) { return a + b; });
  return membership.map(function (a) { return a / sum; });
};

/**
 * Perform one expectation-maximization step and update the GMM weights, means and variances in place.
 * Optionally, if options.variancePrior and options.priorRelevance are defined, mix in the prior.
 * @param {Array} data array of numbers representing the samples to use to update the model
 */
GMM.prototype.updateModel = function (data) {
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
  if (this.options.separationPrior && this.options.separationPriorRelevance) {
    var separationPrior = this.options.separationPrior;
    var priorMeans = _.range(this.nComponents).map(function (a) { return (a * separationPrior); });
    var priorCenter = barycenter(priorMeans, this.weights);
    var center = barycenter(this.means, this.weights);
    for (k = 0; k < this.nComponents; k++) {
      alpha = this.weights[k] / (this.weights[k] + this.options.separationPriorRelevance);
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
    if (this.options.variancePrior && this.options.variancePriorRelevance) {
      alpha = this.weights[k] / (this.weights[k] + this.options.variancePriorRelevance);
      this.vars[k] = alpha * this.vars[k] + (1 - alpha) * this.options.variancePrior;
    }
  }
};

/**
 * Compute the [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood) for the GMM given an array of data.
 * @param {Array} data array of numbers representing the samples to use to update the model
 * @return {Number} the log-likelihood
 */
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
    if (p === 0) {
      return -Infinity;
    } else {
      l += Math.log(p);
    }
  }
  return l;
};

/**
 * Compute the optimal GMM components given an array of data.
 * @param {Array} data array of numbers representing the samples to use to optimize the model
 * @param {Number} [maxIterations=200] maximum number of expectation-maximization steps
 * @param {Number} [logLikelihoodTol=0.0000001] tolerance for the log-likelihood
 * to determine if we reached the optimum
 * @return {Number} the number of steps to reach the converged solution
 * @example
 var gmm = new GMM(3, undefined, [1, 5, 10]);
 var data = [1.2, 1.3, 7.4, 1.4, 14.3, 15.3, 1.0, 7.2];
 gmm.optimize(data); // updates weights, means and variances with the EM algorithm given the data.
 console.log(gmm.means); // >> [1.225, 7.3, 14.8]
 */
GMM.prototype.optimize = function (data, maxIterations, logLikelihoodTol) {
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
  return i;
};

/**
 * Return the model for the GMM as a raw JavaScript Object.
 * @return {Object} the model, with keys `nComponents`, `weights`, `means`, `vars`.
 */
GMM.prototype.model = function () {
  return {
    nComponents: this.nComponents,
    weights: this.weights,
    means: this.means,
    vars: this.vars
  };
};

/**
 * Instantiate a GMM from an Object model and options.
 * @return {GMM} the GMM corresponding to the given model
 * @example var gmm = GMM.fromModel({
    nComponents: 3,
    weights: [0.3, 0.2, 0.5],
    means: [1, 2, 3],
    vars: [1, 1, 0.5]
  });
 */
GMM.fromModel = function (model, options) {
  return new GMM(
    model.nComponents,
    model.weights,
    model.means,
    model.vars,
    options
  );
};
