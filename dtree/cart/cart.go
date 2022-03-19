// package cart implments CART decision tree generator algorithm.
//
// @see https://en.wikipedia.org/wiki/CART_algorithm
//
package cart

import (
	"github.com/gopherd/brain/stat"
	"github.com/gopherd/doge/constraints"
)

func Policy[T constraints.Float](samples []stat.Sample[T], attrs []int) int {
	var bestGini T
	var bestAttr = -1
	var total = T(len(samples))
	for i, attr := range attrs {
		var gini T = 1
		for _, s := range stat.Group(samples, attr) {
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
