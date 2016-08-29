'use strict';

var test = require('tap').test;
var barycenter = require('../utilities/barycenter');

test('barycenter method', function (t) {
  t.plan(5);
  var cropFloat = function (a) { return Number(a.toFixed(5)); };
  t.equals(barycenter([1, 2], [0.5, 0.5]), 1.5);
  t.same(cropFloat(barycenter([1, 2], [0.1, 0.9])), 1.9);
  t.equals(barycenter([1, 2, 3], [0.3, 0.4, 0.3]), 2);
  t.equals(barycenter([1, 2, 3], [3, 4, 3]), 2);
  t.equals(barycenter([1, 2, 3], [0, 0, 0.01]), 3);
});
