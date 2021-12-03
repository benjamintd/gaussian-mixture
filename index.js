'use strict';

// Imports
var gaussian = require('gaussian');

// Constants
var MAX_ITERATIONS = 200;
var EPSILON = 1e-7;

module.exports = GMM;
module.exports.Histogram = Histogram;

/**
 * Instantiate a new GMM.
 * @param {Number} nComponents number of components in the mixture
 * @param {Array} weights array of weights for each component in the mixture, must sum to 1
 * @param {Array} means array of means for each component
 * @param {Array} vars array of variances of each component
 * @param {Object} options an object that can define the `variancePrior`, `separationPrior`, `variancePriorRelevance` and `separationPriorRelevance`.
 * The priors are taken into account when the GMM is optimized given some data. The relevance parameters should be non-negative numbers,
 * 1 meaning that the prior has equal weight as the result of the optimal GMM in each EM step, 0 meaning no influence, and Infinity means a fixed variance (resp. separation).
 * @return {GMM} a gmm object
 * @example var gmm = new GMM(3, [0.3, 0.2, 0.5], [1, 2, 3], [1, 1, 0.5]);
 */
function GMM(nComponents, weights, means, vars, options) {
  this.nComponents = nComponents;
  this.weights = weights === undefined ? range(nComponents).map(function () { return 1 / nComponents; }) : weights;
  this.means = means === undefined ? range(nComponents) : means;
  this.vars = vars === undefined ? range(nComponents).map(function () { return 1; }) : vars;
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
 * @param {Array} gaussians (optional) an Array of length nComponents that contains the gaussians for the GMM
 * @return {Array} (data.length * this.nComponents) matrix with membership weights
 */
GMM.prototype.memberships = function (data, gaussians) {
  var memberships = [];
  if (!gaussians) gaussians = this._gaussians();
  for (var i = 0, n = data.length; i < n; i++) {
    memberships.push(this.membership(data[i], gaussians));
  }
  return memberships;
};

/** @private
 * Given a histogram, determine the membership for each bin and for each component of the GMM.
 * @param {Histogram} h histogram representing the data to find memberships for.
 * @param {Array} gaussians (optional) an Array of length nComponents that contains the gaussians for the GMM
 * @return {Object} a hash from key to memberships, where values are {Array} of length nComponents that sum to 1.
 */
GMM.prototype._membershipsHistogram = function (h, gaussians) {
  var memberships = {};
  if (!gaussians) gaussians = this._gaussians();

  var keys = Object.keys(h.counts);

  for (var i = 0, n = keys.length; i < n; i++) {
    var k = keys[i];
    var v = h.value(k);

    memberships[k] = this.membership(v, gaussians);
  }

  return memberships;
};

/**
 * Given a datapoint, determine its memberships for each component of the GMM.
 * @param {Number} x number representing the sample to score under the model
 * @param {Array} gaussians (optional) an Array of length nComponents that contains the gaussians for the GMM
 * @return {Array} an array of length this.nComponents with membership weights, i.e the probabilities that this datapoint was drawn from the each component
 */
GMM.prototype.membership = function (x, gaussians) {
  var membership = [];
  if (!gaussians) gaussians = this._gaussians();
  var sum = 0;
  for (var i = 0; i < this.nComponents; i++) {
    var m = gaussians[i].pdf(x);
    membership.push(m);
    sum += m;
  }

  return membership.map(function (a) { return a / sum; });
};

/** @private
 * Perform one expectation-maximization step and update the GMM weights, means and variances in place.
 * Optionally, if options.variancePrior and options.priorRelevance are defined, mix in the prior.
 * @param {Array} data array of numbers representing the samples to use to update the model
 * @param {Array} memberships the memberships array for the given data (optional).
 */
GMM.prototype._updateModel = function (data, memberships) {
  // First, we compute the data memberships.
  var n = data.length;
  if (!memberships) memberships = this.memberships(data);
  var alpha;

  // Update the mixture weights
  var componentWeights = [];
  var reduceFunction = function (k) { return function (a, b) { return (a + b[k]); }; };
  for (let k = 0; k < this.nComponents; k++) {
    componentWeights[k] = memberships.reduce(reduceFunction(k), 0);
  }
  this.weights = componentWeights.map(function (a) { return a / n; });

  // Update the mixture means
  for (let k = 0; k < this.nComponents; k++) {
    this.means[k] = 0;
    for (let i = 0; i < n; i++) {
      this.means[k] += memberships[i][k] * data[i];
    }
    this.means[k] /= componentWeights[k];
  }
  // If there is a separation prior:
  if (this.options.separationPrior && this.options.separationPriorRelevance) {
    var separationPrior = this.options.separationPrior;
    var priorMeans = range(this.nComponents).map(function (a) { return (a * separationPrior); });
    var priorCenter = GMM._barycenter(priorMeans, this.weights);
    var center = GMM._barycenter(this.means, this.weights);
    for (let k = 0; k < this.nComponents; k++) {
      alpha = this.weights[k] / (this.weights[k] + this.options.separationPriorRelevance);
      this.means[k] = center + alpha * (this.means[k] - center) + (1 - alpha) * (priorMeans[k] - priorCenter);
    }
  }

  // Update the mixture variances
  for (let k = 0; k < this.nComponents; k++) {
    this.vars[k] = EPSILON; // initialize to some epsilon to avoid zero variance problems.
    for (let i = 0; i < n; i++) {
      this.vars[k] += memberships[i][k] * (data[i] - this.means[k]) * (data[i] - this.means[k]);
    }
    this.vars[k] /= componentWeights[k];
    // If there is a variance prior:
    if (this.options.variancePrior && this.options.variancePriorRelevance) {
      alpha = this.weights[k] / (this.weights[k] + this.options.variancePriorRelevance);
      this.vars[k] = alpha * this.vars[k] + (1 - alpha) * this.options.variancePrior;
    }
  }
};

/** @private
 * Perform one expectation-maximization step and update the GMM weights, means and variances in place.
 * Optionally, if options.variancePrior and options.priorRelevance are defined, mix in the prior.
 * @param {Histogram} h histogram representing the data.
 * @param {Array} memberships the memberships object for the given histogram (optional).
 */
GMM.prototype._updateModelHistogram = function (h, memberships) {
  // First, we compute the data memberships.
  var n = h.total;
  if (!memberships) memberships = this._membershipsHistogram(h);
  var alpha;

  var keys = Object.keys(h.counts);

  // Update the mixture weights
  var componentWeights = [];
  var reduceFunction = function (k) { return function (a, b) { return (a + memberships[b][k] * h.counts[b]); }; };
  for (let k = 0; k < this.nComponents; k++) {
    componentWeights[k] = keys.reduce(reduceFunction(k), 0);
  }
  this.weights = componentWeights.map(function (a) { return a / n; });

  // Update the mixture means
  for (let k = 0; k < this.nComponents; k++) {
    this.means[k] = 0;
    for (let i = 0; i < keys.length; i++) {
      let key = keys[i];
      let v = h.value(key);

      this.means[k] += memberships[key][k] * v * h.counts[key];
    }
    this.means[k] /= componentWeights[k];
  }

  // If there is a separation prior:
  if (this.options.separationPrior && this.options.separationPriorRelevance) {
    var separationPrior = this.options.separationPrior;
    var priorMeans = range(this.nComponents).map(function (a) { return (a * separationPrior); });
    var priorCenter = GMM._barycenter(priorMeans, this.weights);
    var center = GMM._barycenter(this.means, this.weights);
    for (let k = 0; k < this.nComponents; k++) {
      alpha = this.weights[k] / (this.weights[k] + this.options.separationPriorRelevance);
      this.means[k] = center + alpha * (this.means[k] - center) + (1 - alpha) * (priorMeans[k] - priorCenter);
    }
  }

  // Update the mixture variances
  for (let k = 0; k < this.nComponents; k++) {
    this.vars[k] = EPSILON; // initialize to some epsilon to avoid zero variance problems.
    for (let i = 0; i < keys.length; i++) {
      let key = keys[i];
      let v = h.value(key);
      this.vars[k] += memberships[key][k] * (v - this.means[k]) * (v - this.means[k]) * h.counts[key];
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
 * Compute the [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood) for the GMM given data.
 * @param {(Array|Histogram)} data the data array or histogram
 * @return {Number} the log-likelihood
 */
GMM.prototype.logLikelihood = function (data) {
  if (Array.isArray(data)) return this._logLikelihood(data);
  if (Histogram.prototype.isPrototypeOf(data)) return this._logLikelihoodHistogram(data);

  throw new Error('Data must be an Array of a Histogram.');
};

/** @private
 * Compute the [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood) for the GMM given an array of data.
 * @param {Array} data the data array
 * @return {Number} the log-likelihood
 */
GMM.prototype._logLikelihood = function (data) {
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

/** @private
 * Compute the [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood) for the GMM given a histogram.
 * @param {Histogram} h the data histogram
 * @return {Number} the log-likelihood
 */
GMM.prototype._logLikelihoodHistogram = function (h) {
  var l = 0;
  var p = 0;
  var gaussians = this._gaussians();

  var keys = Object.keys(h.counts);

  for (var i = 0, n = keys.length; i < n; i++) {
    p = 0;
    let key = keys[i];
    let v = h.value(key);

    for (var k = 0; k < this.nComponents; k++) {
      p += this.weights[k] * gaussians[k].pdf(v);
    }

    if (p === 0) {
      return -Infinity;
    } else {
      l += Math.log(p) * h.counts[key];
    }
  }
  return l;
};

/** @private
 * Compute the [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood) for the GMM given an array of memberships.
 * @param {Array} memberships the memberships array, matrix of size n * this.nComponents, where n is the size of the data.
 * @return {Number} the log-likelihood
 */
GMM.prototype._logLikelihoodMemberships = function (memberships) {
  var l = 0;
  for (var i = 0, n = memberships.length; i < n; i++) {
    var p = 0;
    for (var k = 0; k < this.nComponents; k++) {
      p += this.weights[k] * memberships[i][k];
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
 * If options has a true flag for `initialize`, the optimization will begin with a K-means++ initialization.
 * This allows to have a data-dependent initialization and should converge quicker and to a better model.
 * The initialization is agnostic to the other priors that the options might contain.
 * @param {(Array|Histogram)} data the data array or histogram
 * @param {Number} [maxIterations=200] maximum number of expectation-maximization steps
 * @param {Number} [logLikelihoodTol=0.0000001] tolerance for the log-likelihood
 * to determine if we reached the optimum
 * @return {Number} the number of steps to reach the converged solution
 */
GMM.prototype.optimize = function (data, maxIterations, logLikelihoodTol) {
  if (Array.isArray(data)) return this._optimize(data, maxIterations, logLikelihoodTol);
  if (Histogram.prototype.isPrototypeOf(data)) return this._optimizeHistogram(data, maxIterations, logLikelihoodTol);

  throw new Error('Data must be an Array of a Histogram.');
};

/** @private
 * Compute the optimal GMM components given an array of data.
 * If options has a true flag for `initialize`, the optimization will begin with a K-means++ initialization.
 * This allows to have a data-dependent initialization and should converge quicker and to a better model.
 * The initialization is agnostic to the other priors that the options might contain.
 * @param {Array} data array of numbers representing the samples to use to optimize the model
 * @param {Number} [maxIterations=200] maximum number of expectation-maximization steps
 * @param {Number} [logLikelihoodTol=0.0000001] tolerance for the log-likelihood
 * to determine if we reached the optimum
 * @return {Number} the number of steps to reach the converged solution
 * @example
 var gmm = new GMM(3, undefined, [1, 5, 10], [1, 1, 1], {initialize: true});
 var data = [1.2, 1.3, 7.4, 1.4, 14.3, 15.3, 1.0, 7.2];
 gmm.optimize(data); // updates weights, means and variances with the EM algorithm given the data.
 console.log(gmm.means); // >> [1.225, 7.3, 14.8]
 */
GMM.prototype._optimize = function (data, maxIterations, logLikelihoodTol) {
  if (this.options.initialize) this._initialize(data);

  maxIterations = maxIterations === undefined ? MAX_ITERATIONS : maxIterations;
  logLikelihoodTol = logLikelihoodTol === undefined ? EPSILON : logLikelihoodTol;
  var logLikelihoodDiff = Infinity;
  var logLikelihood = -Infinity;
  var temp;
  var memberships;
  for (var i = 0; i < maxIterations && logLikelihoodDiff > logLikelihoodTol; i++) {
    this._updateModel(data, memberships);
    memberships = this.memberships(data);
    temp = this._logLikelihoodMemberships(memberships);
    logLikelihoodDiff = Math.abs(logLikelihood - temp);
    logLikelihood = temp;
  }
  return i;
};

/** @private
 * Compute the optimal GMM components given a histogram of data.
 * @param {Histogram} h histogram of data used to optimize the model
 * @param {Number} [maxIterations=200] maximum number of expectation-maximization steps
 * @param {Number} [logLikelihoodTol=0.0000001] tolerance for the log-likelihood
 * to determine if we reached the optimum
 * @return {Number} the number of steps to reach the converged solution
 * @example
 var gmm = new GMM(3, undefined, [1, 5, 10], [1, 1, 1]);
 var h = Histogram.fromData([1.2, 1.3, 7.4, 1.4, 14.3, 15.3, 1.0, 7.2]);
 gmm.optimizeHistogram(h); // updates weights, means and variances with the EM algorithm given the data.
 console.log(gmm.means); // >> [1.225, 7.3, 14.8]
 */
GMM.prototype._optimizeHistogram = function (h, maxIterations, logLikelihoodTol) {
  if (this.options.initialize) this._initializeHistogram(h);

  maxIterations = maxIterations === undefined ? MAX_ITERATIONS : maxIterations;
  logLikelihoodTol = logLikelihoodTol === undefined ? EPSILON : logLikelihoodTol;
  var logLikelihoodDiff = Infinity;
  var logLikelihood = -Infinity;
  var temp;
  for (var i = 0; i < maxIterations && logLikelihoodDiff > logLikelihoodTol; i++) {
    this._updateModelHistogram(h);
    temp = this._logLikelihoodHistogram(h);
    logLikelihoodDiff = Math.abs(logLikelihood - temp);
    logLikelihood = temp;
  }
  return i;
};


/** @private
 * Initialize the GMM given data with the [K-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) initialization algorithm.
 * The k-means++ algorithm choses datapoints amongst the data at random, while ensuring that the chosen seeds are far from each other.
 * The resulting seeds are returned sorted.
 * @param {Array} data array of numbers representing the samples to use to optimize the model
 * @return {Array} an array of length nComponents that contains the means for the initialization.
 * @example
 var gmm = new GMM(3, [0.3, .04, 0.3], [1, 5, 10]);
 var data = [1.2, 1.3, 7.4, 1.4, 14.3, 15.3, 1.0, 7.2];
 gmm.initialize(data); // updates the means of the GMM with the K-means++ initialization algorithm, returns something like [1.3, 7.4, 14.3]
 */
GMM.prototype._initialize = function (data) {
  var n = data.length;

  if (n < this.nComponents) throw new Error('Data must have more points than the number of components in the model.');

  var means = [];

  // Find the first seed at random
  means.push(data[Math.round(Math.random() * (n - 1))]);

  var distances = [];

  // Chose all other seeds
  for (var m = 1; m < this.nComponents; m++) {
    // Compute the distance from each datapoint
    var dsum = 0;
    for (var i = 0; i < n; i++) {
      var meansDistances = means.map(function (x) { return (x - data[i]) * (x - data[i]); });
      var d = meansDistances.reduce(function (a, b) { return Math.min(a, b); });
      distances[i] = d;
      dsum += d;
    }

    // Chose the next seed at random with probabilities d / dsum
    var r = Math.random();
    var c;
    for (var j = 0; j < n; j++) {
      var p = (distances[j] / dsum) || 0;
      if (p > r || j === (n - 1)) {
        c = data[j];
        break;
      } else {
        r -= p;
      }
    }

    means.push(c);
  }

  means.sort(function (a, b) { return a - b; });
  this.means = means;
  return means;
};

/** @private
 * Initialize the GMM given data with the [K-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) initialization algorithm.
 * The k-means++ algorithm choses datapoints amongst the data at random, while ensuring that the chosen seeds are far from each other.
 * The resulting seeds are returned sorted.
 * @param {Array} data array of numbers representing the samples to use to optimize the model
 * @return {Array} an array of length nComponents that contains the means for the initialization.
 * @example
 var gmm = new GMM(3, [0.3, .04, 0.3], [1, 5, 10]);
 var data = [1.2, 1.3, 7.4, 1.4, 14.3, 15.3, 1.0, 7.2];
 gmm.initialize(data); // updates the means of the GMM with the K-means++ initialization algorithm, returns something like [1.3, 7.4, 14.3]
 */
GMM.prototype._initializeHistogram = function (h) {
  var n = h.total;

  if (n < this.nComponents) throw new Error('Data must have more points than the number of components in the model.');

  var keys = Object.keys(h.counts);
  var means = [];

  // Find the first seed at random
  var r = Math.random();

  for (let i = 0; i < keys.length; i++) {
    let k = keys[i];
    let p = (h.counts[k] / n) || 0;
    if (p > r || i === (keys.length - 1)) {
      means.push(h.value(k));
      break;
    } else {
      r -= p;
    }
  }

  var distances = [];

  // Chose all other seeds
  for (let m = 1; m < this.nComponents; m++) {

    // Compute the distance from each datapoint
    var dsum = 0;
    for (let i = 0; i < keys.length; i++) {
      let k = keys[i];
      var meansDistances = means.map(function (x) { return (x - h.value(k)) * (x - h.value(k)); });
      var d = meansDistances.reduce(function (a, b) { return Math.min(a, b); });
      distances[i] = d * h.counts[k];
      dsum += d;
    }

    // Chose the next seed at random with probabilities d / dsum
    let r = Math.random();

    for (let i = 0; i < keys.length; i++) {
      let k = keys[i];
      let p = (distances[i] / dsum) || 0;
      if (p > r || i === (keys.length - 1)) {
        means.push(h.value(k));
        break;
      } else {
        r -= p;
      }
    }
  }

  means.sort(function (a, b) { return a - b; });
  this.means = means;
  return means;
};

/** @private
 * Compute the barycenter given an array and weights.
 * @param {Array} array the array of values to find the barycenter from
 * @param {Array} weights an array of same length that contains the weight for each value
 * @return {Number} the barycenter
 */
GMM._barycenter = function (array, weights) {
  var total = 0;
  var barycenter = 0;
  for (var i = 0, n = array.length; i < n; i++) {
    total += weights[i];
    barycenter += array[i] * weights[i];
  }
  return barycenter / total;
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

/**
 * Instantiate a new Histogram.
 * @param {Object} [h={}] an object with keys 'counts' and 'bins'. Both are optional.
 * An observation x will be counted for the key i if bins[i][0] <= x < bins[i][1].
 * If bins are not specified, the bins will be corresponding to one unit in the scale of the data.
 * The keys of the 'counts' hash will be stringified integers.
 * @return {Histogram} a histogram object.
 * It has keys 'bins' (possibly null) and 'counts'.
 * @example var h = new Histogram({counts: {'a': 3, 'b': 2, 'c': 5}, bins: {'a': [0, 2], 'b': [2, 4], 'c': [4, 7]}});
 * @example var h = new Histogram({counts: {'1': 3, '2': 2, '3': 5}});
 * @example var h = new Histogram();
 */
function Histogram(h) {
  h = h || {};
  this.bins = h.bins || null;
  this.counts = h.counts || {};
  this.total = Histogram._total(h);
}

/** @private
 * Get the key corresponding to a single element.
 * @param {Array} x observation to classify in the histogram
 * @param {Object} [bins=undefined] a map from key to range (a range being an array of two elements)
 * An observation x will be counted for the key i if `bins[i][0] <= x < bins[i][1]`.
 * If not specified, the bins will be corresponding to one unit in the scale of the data.
 * @return {String} the key to add the observation in the histogram
 */
Histogram._classify = function (x, bins) {
  if (bins === null || bins === undefined) return Math.round(x).toString();

  var keys = Object.keys(bins);

  for (var i = 0, n = keys.length; i < n; i++) {
    var bounds = bins[keys[i]];
    if (bounds[0] <= x && x < bounds[1]) return keys[i];
  }

  return null;
};

/** @private
 * Get the total count for the histogram
 * @param {Histogram} h a histogram
 * @return {Number} the total count for the histogram
 */
Histogram._total = function (h) {
  return Object.keys(h.counts)
    .map(function (k) { return h.counts[k]; })
    .reduce(function (a, b) { return a + b; }, 0);
};

/**
 * Add an observation to an histogram.
 * @param {Array} x observation to add tos the histogram
 * @return {Histogram} the histogram with added value.
 */
Histogram.prototype.add = function (x) {
  var c = Histogram._classify(x, this.bins);
  if (c !== null) {
    if (!this.counts[c]) this.counts[c] = 1;
    else this.counts[c] += 1;

    this.total += 1;
  }

  return this;
};

/**
 * Return a data array from a histogram.
 * @return {Array} an array of observations derived from the histogram counts.
 */
Histogram.prototype.flatten = function () {
  var r = [];

  var keys = Object.keys(this.counts);

  for (var i = 0, n = keys.length; i < n; i++) {
    var k = keys[i];
    var v = this.value(k);

    for (var j = 0; j < this.counts[k]; j++) {
      r.push(v);
    }
  }

  return r;
};

/**
 * Instantiate a new Histogram.
 * @param {Array} [data=[]] array of observations to include in the histogram.
 * Observations that do not correspond to any bin will be discarded.
 * @param {Object} [bins={}] a map from key to range (a range being an array of two elements)
 * An observation x will be counted for the key i if `bins[i][0] <= x < bins[i][1]`.
 * If not specified, the bins will be corresponding to one unit in the scale of the data.
 * @return {Histogram} a histogram object
 * It has keys 'bins' (possibly null) and 'counts'.
 * @example var h = Histogram.fromData([1, 2, 2, 2, 5, 5], {A: [0, 1], B: [1, 5], C: [5, 10]});
 // {bins: {A: [0, 1], B: [1, 5], C: [5, 10]}, counts: {A: 0, B: 4, C: 2}}
 * @example var h = Histogram.fromData([1, 2, 2, 2, 2.4, 2.5, 5, 5]);
 // {counts: {'1': 1, '2': 4, '3': 1, '5': 2}}
 */
Histogram.fromData = function (data, bins) {
  var h = new Histogram({bins: bins, counts: {}});

  for (var i = 0, n = data.length; i < n; i++) {
    h.add(data[i]);
  }

  return h;
};

/**
 * Return the median value for the given key, derived from the bins.
 * @return {Number} the value for the provided key.
 */
Histogram.prototype.value = function (key) {
  if (this.bins) {
    if (this.bins[key]) return (this.bins[key][0] + this.bins[key][1]) / 2;
    else throw new Error('No bin for this key.');
  } else {
    return Number(key);
  }
};

function range(n) {
  var result = new Array(n);
  for (var i = 0; i < n; i++) {
    result[i] = i;
  }
  return result;
}
