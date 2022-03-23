// package cart implments CART decision tree generator algorithm.
//
// @see https://en.wikipedia.org/wiki/CART_algorithm
//
package cart

import (
	"github.com/gopherd/ml/model"
	"github.com/gopherd/doge/constraints"
)

func Policy[T constraints.Float](samples []model.Sample[T], attrs []int) int {
	var bestGini T
	var bestAttr = -1
	var total = T(len(samples))
	for i, attr := range attrs {
		var gini T = 1
		for _, s := range model.Group(samples, attr) {
			var p = T(len(s)) / total
			gini -= p * p
		}
		if i == 0 || gini < bestGini {
			bestGini = gini
			bestAttr = i
		}
	}
	return bestAttr
}
