var test = require('tap').test
var GMM = require('../index')

test('Initialization of a new GMM object.', function (t) {
  t.plan(2)

  function f (k) {
    return function () {
      var gmm = new GMM(3, k)
      gmm.sample(0)
    }
  }
  t.throws(f([1, 2]))
  t.doesNotThrow(f([1, 2, 3]))
})

test('Random sampling.', function (t) {
  t.plan(1)

  var gmm = new GMM(3)
  t.equals(5, gmm.sample(5).length)
})

test('Gaussians of a mixture model.', function (t) {
  t.plan(6)

  var gmm = new GMM(3, [1 / 3, 1 / 3, 1 / 3], [0, 10, 20], [1, 2, 0.5])
  var gaussians = gmm._gaussians()
  t.equals(gaussians[0].mean, 0)
  t.equals(gaussians[1].mean, 10)
  t.equals(gaussians[2].mean, 20)
  t.equals(gaussians[0].variance, 1)
  t.equals(gaussians[1].variance, 2)
  t.equals(gaussians[2].variance, 0.5)
})

test('Computing membership for one datapoint', function (t) {
  t.plan(2)

  var gmm = new GMM(3, undefined, [0, 10, 20])
  t.equals(gmm.membership(5)[0], gmm.membership(5)[1])
  t.equals(gmm.membership(0)[0] > 0.99, true)
})

test('Shape of the membership matrix', function (t) {
  t.plan(2)

  var gmm = new GMM(5)
  var memberships = gmm.memberships([1, 2, 3, 4, 5, 6])
  t.equals(memberships.length, 6)
  t.equals(memberships[0].length, 5)
})

test('Convergence of model update', function (t) {
  t.plan(9)

  var refGmm = new GMM(3, [0.2, 0.5, 0.3], [0, 10, 30], [1, 2, 4])
  var testGmm = new GMM(3, undefined, [-1, 13, 25], [1, 1, 1])
  var data = require('./fixtures/data.json') // 200 samples from refGmm
  for (var i = 0; i < 200; i++) {
    testGmm.updateModel(data)
  }
  for (var j = 0; j < 3; j++) {
    t.equals(Math.abs(testGmm.weights[j] - refGmm.weights[j]) < 0.1, true)
    t.equals(Math.abs(testGmm.means[j] - refGmm.means[j]) < 1, true)
    t.equals(Math.abs(testGmm.vars[j] - refGmm.vars[j]) < 1, true)
  }
})
