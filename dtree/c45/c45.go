// package c45 implments C4.5 decision tree generator algorithm.
//
// C4.5 is an algorithm used to generate a decision tree developed by Ross Quinlan.
// @see https://en.wikipedia.org/wiki/C4.5_algorithm
//
package c45

import (
	"github.com/gopherd/brain/stat"
	"github.com/gopherd/doge/constraints"
)

func Policy[T constraints.Float](samples []stat.Sample[T], attrs []int) int {
	var bestGain float64
	var bestAttr = -1
	var total = float64(len(samples))
	for i, attr := range attrs {
		var sum float64
		for _, s := range stat.Group(samples, attr) {
			sum += float64(len(s)) / total * stat.EntropySet(s)
		}
		if i == 0 || sum < bestGain {
			bestGain = sum
			bestAttr = i
		}
	}
	return bestAttr
}
