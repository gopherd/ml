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
	var bestGini float64
	var bestAttr = -1
	var total = float64(len(samples))
	for i, attr := range attrs {
		var gini = 1.0
		for _, s := range stat.Group(samples, attr) {
			var p = float64(len(s)) / total
			gini -= p * p
		}
		if i == 0 || gini < bestGini {
			bestGini = gini
			bestAttr = i
		}
	}
	return bestAttr
}
