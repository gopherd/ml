// package id3 implments ID3 decision tree generator algorithm.
//
// ID3 is an algorithm used to generate a decision tree developed by Ross Quinlan.
// @see https://en.wikipedia.org/wiki/ID3_algorithm
//
package id3

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
			sum += float64(len(s)) / total * stat.SumEntropySet(s)
		}
		if i == 0 || sum < bestGain {
			bestGain = sum
			bestAttr = i
		}
	}
	return bestAttr
}
