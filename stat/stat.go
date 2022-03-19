package stat

import (
	"math"
	"math/bits"

	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/container/maps"
	"github.com/gopherd/doge/container/slices"
	"github.com/gopherd/doge/math/functional"
	"github.com/gopherd/doge/math/tensor"
)

const Epsilon = 1e-6

type Sample[T constraints.Float] struct {
	Attributes tensor.Vector[T]
	Weight     T
	Label      T
}

// Counters counters number of each class
func Counters[S ~[]Sample[T], T constraints.Float](samples S) map[T]int {
	if len(samples) == 0 {
		return nil
	}
	var counters = make(map[T]int)
	for i := range samples {
		counters[samples[i].Label]++
	}
	return counters
}

// Entropy computes information entropy for p
func Entropy[T constraints.Float](p T) T {
	if p < Epsilon {
		return 0
	}
	return -p * T(math.Log2(float64(p)))
}

// SumEntropy computes information entropy by probablities
func SumEntropy[S ~[]T, T constraints.Float](probs S) T {
	return slices.SumFunc[S, func(T) T, T, T](probs, Entropy[T])
}

// SumEntropySet computes information entropy of set
func SumEntropySet[S ~[]Sample[T], T constraints.Float](samples S) T {
	var counters = make(map[T]T)
	if len(samples) == 0 {
		return 0
	}
	for i := range samples {
		counters[samples[i].Label]++
	}
	var total = T(len(samples))
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

// Gini computes gini index by probablities
//
//	g = 1 - Σk(pk^2)
func Gini[S ~[]T, T constraints.Float](probs S) T {
	return 1 - slices.SumFunc[S, func(T) T, T, T](probs, functional.Square[T])
}

func Log2(n uint) int {
	if n < 1 {
		return 0
	}
	return bits.UintSize - bits.LeadingZeros(n-1)
}
