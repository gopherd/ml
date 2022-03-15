package stat

import (
	"math"

	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/container/maps"
	"github.com/gopherd/doge/container/slices"
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

// Entropy computes information entropy for p
func Entropy[T constraints.Float](p T) float64 {
	if p < Epsilon {
		return 0
	}
	return -float64(p) * math.Log2(float64(p))
}

// SumEntropy computes information entropy by probablities
func SumEntropy[S ~[]T, T constraints.Float](probs S) float64 {
	return slices.SumFunc[S, func(T) float64, T, float64](probs, Entropy[T])
}

// SumEntropySet computes information entropy of set
func SumEntropySet[S ~[]Sample[T], T constraints.SignedReal](samples S) float64 {
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
	return SumEntropy(probs)
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

func square64[T constraints.Float](x T) float64 {
	return float64(x) * float64(x)
}

// Gini computes gini index by probablities
//
//	g = 1 - Σk(pk^2)
func Gini[S ~[]T, T constraints.Float](probs S) float64 {
	return 1 - slices.SumFunc[S, func(T) float64, T, float64](probs, square64[T])
}
