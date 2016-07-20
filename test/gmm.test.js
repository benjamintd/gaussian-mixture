var test = require('tap').test
var GMM = require('../index')

test('random sample', function (t) {
  var gmm = new GMM(3)
  t.plan(1)
  t.equals(5, gmm.sample(5).length)
})
