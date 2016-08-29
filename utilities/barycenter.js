'use strict';

module.exports = function (array, weights) {
  var total = 0;
  var barycenter = 0;
  for (var i = 0, n = array.length; i < n; i++) {
    total += weights[i];
    barycenter += array[i] * weights[i];
  }
  return barycenter / total;
};
