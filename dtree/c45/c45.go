// package c45 implments C4.5 decision tree generator algorithm.
//
// C4.5 is an algorithm used to generate a decision tree developed by Ross Quinlan.
// @see https://en.wikipedia.org/wiki/C4.5_algorithm
//
package c45

import (
	"github.com/gopherd/ml/model"
	"github.com/gopherd/doge/constraints"
)

func Policy[T constraints.Float](samples []model.Sample[T], attrs []int) int {
	var bestGain T
	var bestAttr = -1
	var total = T(len(samples))
	var ent = model.SumEntropySet(samples)
	for i, attr := range attrs {
		var sum = ent
		var iv T
		for _, s := range model.Group(samples, attr) {
			sum -= T(len(s)) / total * model.SumEntropySet(s)
			var p = T(len(s)) / total
			iv += model.Entropy(p)
		}
		if iv < model.Epsilon {
			return i
		}
		sum /= iv
		if i == 0 || sum > bestGain {
			bestGain = sum
			bestAttr = i
		}
	}
	return bestAttr
}
