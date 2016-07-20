var test = require('tap').test
var GMM = require('../index')

test('initialization', function (t) {
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

test('random sample', function (t) {
  t.plan(1)

  var gmm = new GMM(3)
  t.equals(5, gmm.sample(5).length)
})

test('gaussians', function (t) {
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

test('scoring one data point', function (t) {
  t.plan(3)

  var gmm = new GMM(3, undefined, [0, 10, 20])
  t.equals(gmm.scoreSample(5)[0], 0.5)
  t.equals(gmm.scoreSample(5)[1], 0.5)
  t.equals(gmm.scoreSample(0)[0], 1.0)
})
