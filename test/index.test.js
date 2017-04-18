'use strict';

var test = require('tap').test;
var data = require('./fixtures/data.json'); // 200 samples from refGmm

var GMM = require('../index');
var Histogram = require('../index').Histogram;

test('Initialization of a new GMM object.', function (t) {
  t.plan(2);

  function f(k) {
    return function () {
      var gmm = new GMM(3, k);
      gmm.sample(0);
    };
  }
  t.throws(f([1, 2]));
  t.doesNotThrow(f([1, 2, 3]));
});

test('Random sampling.', function (t) {
  t.plan(1);

  var gmm = new GMM(3);
  t.equals(5, gmm.sample(5).length);
});

test('Gaussians of a mixture model.', function (t) {
  t.plan(6);

  var gmm = new GMM(3, [1 / 3, 1 / 3, 1 / 3], [0, 10, 20], [1, 2, 0.5]);
  var gaussians = gmm._gaussians();
  t.equals(gaussians[0].mean, 0);
  t.equals(gaussians[1].mean, 10);
  t.equals(gaussians[2].mean, 20);
  t.equals(gaussians[0].variance, 1);
  t.equals(gaussians[1].variance, 2);
  t.equals(gaussians[2].variance, 0.5);
});

test('Computing membership for one datapoint', function (t) {
  t.plan(2);

  var gmm = new GMM(3, undefined, [0, 10, 20]);
  t.equals(gmm.membership(5)[0], gmm.membership(5)[1]);
  t.equals(gmm.membership(0)[0] > 0.99, true);
});

test('Shape of the membership matrix', function (t) {
  t.plan(2);

  var gmm = new GMM(5);
  var memberships = gmm.memberships([1, 2, 3, 4, 5, 6]);
  t.equals(memberships.length, 6);
  t.equals(memberships[0].length, 5);
});

test('Convergence of model update', function (t) {
  t.plan(9);

  var refGmm = new GMM(3, [0.2, 0.5, 0.3], [0, 10, 30], [1, 2, 4]);
  var testGmm = new GMM(3, undefined, [-1, 13, 25], [1, 1, 1]);
  for (var i = 0; i < 200; i++) {
    testGmm._updateModel(data);
  }
  for (var j = 0; j < 3; j++) {
    t.equals(Math.abs(testGmm.weights[j] - refGmm.weights[j]) < 0.1, true);
    t.equals(Math.abs(testGmm.means[j] - refGmm.means[j]) < 1, true);
    t.equals(Math.abs(testGmm.vars[j] - refGmm.vars[j]) < 1, true);
  }
});

test('log likelihood function', function (t) {
  t.plan(15);

  var gmm = new GMM(3, undefined, [-1, 15, 37], [1, 1, 1]);
  var l = -Infinity;
  var temp;
  for (var i = 0; i < 15; i++) {
    gmm._updateModel(data);
    temp = gmm.logLikelihood(data);
    t.equals(temp - l >= -1e-5, true);
    l = temp;
  }
});

test('EM full optimization', function (t) {
  t.plan(11);

  var gmm = new GMM(3, undefined, [-1, 13, 25], [1, 1, 1]);
  var gmm2 = new GMM(3, undefined, [-1, 13, 25], [1, 1, 1]);

  for (var i = 0; i < 200; i++) {
    gmm._updateModel(data);
  }
  var counter = gmm2.optimize(data);

  t.equals(counter, 3);
  for (var j = 0; j < 3; j++) {
    t.equals(Math.abs(gmm.weights[j] - gmm2.weights[j]) < 1e-7, true);
    t.equals(Math.abs(gmm.means[j] - gmm2.means[j]) < 1e-7, true);
    t.equals(Math.abs(gmm.vars[j] - gmm2.vars[j]) < 1e-7, true);
  }

  var gmm3 = new GMM(3, [0.4, 0.3, 0.4], [-10, -5, -1], [1, 1, 1], {initialize: true});
  gmm3.optimize([1, 2, 3, 1, 2, 3, 1, 1, 2, 2, 3, 3]);
  t.same(gmm3.means, [1, 2, 3]);
});

test('Variance prior', function (t) {
  t.plan(3);

  var options = {
    variancePrior: 3,
    variancePriorRelevance: 0.01
  };
  var options2 = {
    variancePrior: 3,
    variancePriorRelevance: 1
  };
  var options3 = {
    variancePrior: 3,
    variancePriorRelevance: 1000000
  };
  var gmm = new GMM(3, undefined, [-1, 13, 25], [1, 1, 1], options);
  var gmm2 = new GMM(3, undefined, [-1, 13, 25], [1, 1, 1], options2);
  var gmm3 = new GMM(3, undefined, [-1, 13, 25], [1, 1, 1], options3);
  gmm.optimize(data);
  gmm2.optimize(data);
  gmm3.optimize(data);

  var cropFloat = function (a) { return Number(a.toFixed(1)); };
  t.same(gmm.vars.map(cropFloat), [1.1, 2.5, 4.6]);
  t.same(gmm2.vars.map(cropFloat), [2.7, 2.8, 3.4]);
  t.same(gmm3.vars.map(cropFloat), [3, 3, 3]);
});

test('Separation prior', function (t) {
  t.plan(3);

  var options = {
    separationPrior: 3,
    separationPriorRelevance: 0.01
  };
  var options2 = {
    separationPrior: 3,
    separationPriorRelevance: 1
  };
  var options3 = {
    separationPrior: 3,
    separationPriorRelevance: 1000000
  };
  var gmm = new GMM(3, undefined, [-1, 13, 25], [1, 1, 1], options);
  var gmm2 = new GMM(3, undefined, [-1, 13, 25], [1, 1, 1], options2);
  var gmm3 = new GMM(3, undefined, [-1, 13, 25], [1, 1, 1], options3);
  gmm.optimize(data);
  gmm2.optimize(data);
  gmm3.optimize(data);

  var cropFloat = function (a) { return Number(a.toFixed(1)); };
  t.same(gmm.means.map(cropFloat), [0.7, 10.2, 29.8]);
  t.same(gmm2.means.map(cropFloat), [11.6, 13.5, 17.8]);
  t.same(gmm3.means.map(cropFloat), [11.4, 14.4, 17.4]);
});

test('Model', function (t) {
  t.plan(2);

  var gmm = new GMM(3, [0.4, 0.2, 0.4], [-1, 13, 25], [1, 2, 1]);

  var model = {
    nComponents: 3,
    weights: [0.4, 0.2, 0.4],
    means: [-1, 13, 25],
    vars: [1, 2, 1]
  };

  t.same(gmm.model(), model);
  t.same(GMM.fromModel(model), gmm);
});

test('Barycenter method', function (t) {
  t.plan(5);
  var cropFloat = function (a) { return Number(a.toFixed(5)); };
  t.equals(GMM._barycenter([1, 2], [0.5, 0.5]), 1.5);
  t.same(cropFloat(GMM._barycenter([1, 2], [0.1, 0.9])), 1.9);
  t.equals(GMM._barycenter([1, 2, 3], [0.3, 0.4, 0.3]), 2);
  t.equals(GMM._barycenter([1, 2, 3], [3, 4, 3]), 2);
  t.equals(GMM._barycenter([1, 2, 3], [0, 0, 0.01]), 3);
});

test('Km++ Initialization', function (t) {
  var gmm = new GMM(3, [0.4, 0.2, 0.4], [-1, 13, 25], [1, 2, 1]);

  var means = gmm.initialize([1, 3, 3, 3, 2, 2, 1, 1, 3, 2, 2, 1, 3, 3, 3, 2, 1]);
  t.same(means, [1, 2, 3]);

  t.same(gmm.initialize([1, 1, 1, 1]), [1, 1, 1]);
  t.same(gmm.initialize([1, 1, 1, 2, 17]), [1, 2, 17]);

  t.throws(function () { gmm.initialize([1]); }, new Error('Data must have more points than the number of components in the model.'));

  t.end();
});

test('memberships - histogram', function (t) {
  var h = Histogram.fromData([1, 2, 5, 5.4, 5.5, 6, 7, 7]);
  var gmm = GMM.fromModel({
    means: [1, 5, 7],
    vars: [2, 2, 2],
    weights: [0.3, 0.5, 0.2],
    nComponents: 3
  });

  t.same(gmm._membershipsHistogram(h), {
    1: [0.9818947940807183, 0.01798403047511045, 0.00012117544417123207],
    2: [0.8788782427321509, 0.11894323591065209, 0.0021785213571970234],
    5: [0.013212886953789417, 0.7213991842739687, 0.265387928772242],
    6: [0.0012378419366357771, 0.49938107903168216, 0.49938107903168216],
    7: [0.00009021165708731931, 0.268917159718714, 0.7309926286241988]
  });

  t.end();
});

test('log likelihood - histogram', function (t) {
  var h = Histogram.fromData([1, 2, 5, 5, 5, 6, 7, 7]);
  var gmm = GMM.fromModel({
    means: [1, 5, 7],
    vars: [2, 2, 2],
    weights: [0.3, 0.5, 0.2],
    nComponents: 3
  });

  t.equal(gmm.logLikelihood(h), gmm.logLikelihood([1, 2, 5, 5, 5, 6, 7, 7]));
  t.end();
});

test('optimize - histogram', function (t) {
  var h = Histogram.fromData([1, 2, 5, 5, 5, 6, 7, 7]);
  var gmm = GMM.fromModel({
    means: [1, 5, 7],
    vars: [2, 2, 2],
    weights: [0.3, 0.5, 0.2],
    nComponents: 3
  });
  var gmm2 = GMM.fromModel({
    means: [1, 5, 7],
    vars: [2, 2, 2],
    weights: [0.3, 0.5, 0.2],
    nComponents: 3
  });
  gmm._optimizeHistogram(h);
  gmm2._optimize([1, 2, 5, 5, 5, 6, 7, 7]);
  var round = x => Number(x.toFixed(5));
  t.same(gmm.model().means.map(round), gmm2.model().means.map(round));
  t.same(gmm.model().vars.map(round), gmm2.model().vars.map(round));
  t.same(gmm.model().weights.map(round), gmm2.model().weights.map(round));

  var options = {
    variancePrior: 3,
    variancePriorRelevance: 0.5,
    separationPrior: 3,
    separationPriorRelevance: 1
  };

  gmm.options = options;
  gmm2.options = options;

  gmm.optimize(h);
  gmm2._optimize([1, 2, 5, 5, 5, 6, 7, 7]);
  t.same(gmm.model().means.map(round), gmm2.model().means.map(round));
  t.same(gmm.model().vars.map(round), gmm2.model().vars.map(round));
  t.same(gmm.model().weights.map(round), gmm2.model().weights.map(round));

  t.end();
});

test('histogram total', function (t) {
  var d = [1, 2, 3, 4, 5, 5, 6, 6, 6];

  var h = Histogram.fromData(d);

  t.equals(Histogram._total(h), 9);
  t.end();
});

test('histogram classify', function (t) {
  t.equals(Histogram._classify(3.4), '3');
  t.equals(Histogram._classify(3.4, {'A': [1, 2], 'B': [3, 3.4], 'C': [3.4, 5], 'D': [5, 6]}), 'C');
  t.same(Histogram._classify(7, {'A': [1, 2], 'B': [3, 3.4], 'C': [3.4, 5], 'D': [5, 6]}), null);
  t.end();
});

test('histogram value', function (t) {
  var h = new Histogram({
    bins: {'A': [1, 2], 'B': [3, 3.4], 'C': [3.4, 5], 'D': [5, 6]},
    counts: {'A': 5, 'B': 3}
  });
  t.equals(h.value('A'), 1.5);
  t.equals(h.value('B'), 3.2);
  t.true(isNaN(h.value('E')));
  t.end();
});

test('histogram flatten', function (t) {
  var h = new Histogram({
    bins: {'A': [1, 2], 'B': [3, 3.4], 'C': [3.4, 5], 'D': [5, 6]},
    counts: {'A': 3, 'B': 2}
  });
  t.same(h.flatten(), [1.5, 1.5, 1.5, 3.2, 3.2]);
  t.end();
});
