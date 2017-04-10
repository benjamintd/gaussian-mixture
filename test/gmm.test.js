'use strict';

var test = require('tap').test;
var data = require('./fixtures/data.json'); // 200 samples from refGmm

var GMM = require('../index');

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
    testGmm.updateModel(data);
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
    gmm.updateModel(data);
    temp = gmm.logLikelihood(data);
    t.equals(temp - l >= -1e-5, true);
    l = temp;
  }
});

test('EM full optimization', function (t) {
  t.plan(10);

  var gmm = new GMM(3, undefined, [-1, 13, 25], [1, 1, 1]);
  var gmm2 = new GMM(3, undefined, [-1, 13, 25], [1, 1, 1]);

  for (var i = 0; i < 200; i++) {
    gmm.updateModel(data);
  }
  var counter = gmm2.optimize(data);

  t.equals(counter, 3);
  for (var j = 0; j < 3; j++) {
    t.equals(Math.abs(gmm.weights[j] - gmm2.weights[j]) < 1e-7, true);
    t.equals(Math.abs(gmm.means[j] - gmm2.means[j]) < 1e-7, true);
    t.equals(Math.abs(gmm.vars[j] - gmm2.vars[j]) < 1e-7, true);
  }
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
