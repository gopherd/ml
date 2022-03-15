package stat

import (
	"math"

	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/container/maps"
	"github.com/gopherd/doge/math/tensor"
)

const Epsilon = 1e-6

type Sample[T constraints.SignedReal] struct {
	Attributes tensor.Vector[T]
	Class      int
}

// Counters counters number of each class
func Counters[S ~[]Sample[T], T constraints.SignedReal](samples S) map[int]int {
	if len(samples) == 0 {
		return nil
	}
	var counters = make(map[int]int)
	for i := range samples {
		counters[samples[i].Class]++
	}
	return counters
}

// Entropy computes information entropy by probablities
func Entropy[S ~[]T, T constraints.Float](probs S) T {
	var sum float64
	for _, p := range probs {
		if p < Epsilon {
			continue
		}
		sum -= float64(p) * math.Log2(float64(p))
	}
	return T(sum)
}

// EntropySet computes information entropy of set
func EntropySet[S ~[]Sample[T], T constraints.SignedReal](samples S) float64 {
	var counters = make(map[int]float64)
	if len(samples) == 0 {
		return 0
	}
	for i := range samples {
		counters[samples[i].Class]++
	}
	var total = float64(len(samples))
	var probs = maps.Values(counters)
	for i := range probs {
		probs[i] /= total
	}
	return Entropy(probs)
}

// Group groups samples by attribute
func Group[S ~[]Sample[T], T constraints.Float](samples S, attribute int) map[T]S {
	var m = make(map[T]S)
	for _, x := range samples {
		var attr = x.Attributes[attribute]
		m[attr] = append(m[attr], x)
	}
	return m
}
