// package c45 implments C4.5 decision tree generator algorithm.
//
// C4.5 is an algorithm used to generate a decision tree developed by Ross Quinlan.
// @see https://en.wikipedia.org/wiki/C4.5_algorithm
//
package c45

import (
	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/container/pair"
	"github.com/gopherd/ml/model"
)

func Policy[T constraints.Float](samples []model.Sample[T], attrs []int) int {
	var total = T(len(samples))
	var ent = model.SumEntropySet(samples)

	// calculate gain and iv for each attribute
	var gainAndIVs = make([]pair.Pair[T, T], len(attrs))
	var avgGain, maxGain T
	for i, attr := range attrs {
		var gain = ent
		var iv T
		for _, s := range model.Group(samples, attr) {
			var p = T(len(s)) / total
			gain -= p * model.SumEntropySet(s)
			iv += model.Entropy(p)
		}
		if iv < model.Epsilon {
			return i
		}
		gainAndIVs[i].First = gain
		gainAndIVs[i].Second = iv
		avgGain += gain
		if i == 0 || gain > maxGain {
			maxGain = gain
		}
	}
	avgGain /= total
	var filter = maxGain > avgGain

	// select attribute which has maxinum gain ratio from where
	// gain greater or equal to the average gain.
	var bestRatio T
	var bestAttr = -1
	for i := range gainAndIVs {
		if filter && gainAndIVs[i].First < avgGain {
			continue
		}
		var ratio = gainAndIVs[i].First / gainAndIVs[i].Second
		if bestAttr < 0 || ratio > bestRatio {
			bestAttr = i
			bestRatio = ratio
		}
	}
	return bestAttr
}
