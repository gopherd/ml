package random

import (
	"math"
	"math/rand"
	"sort"

	"github.com/gopherd/brain/model"
	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/container/slices"
	"github.com/gopherd/doge/math/tensor"
)

type Distribution[T constraints.Float] interface {
	Generate() model.Sample[T]
}

func random[T constraints.Float](total T, accWeights []T) int {
	var p = T(rand.Float64()) * total
	var n = len(accWeights)
	var i = sort.Search(n, func(i int) bool {
		return accWeights[i] > p
	})
	if i == n {
		i = rand.Intn(n)
	}
	return i
}

type gaussian[T constraints.Float] struct {
	// u = E(X)
	// s = Σ⁻¹
	// f(x) = const * exp(-0.5(x-u)ᵀ·s·(x-u)
	u tensor.Vector[T]
	s tensor.Matrix[T]

	total    T
	shape    tensor.Indices
	weights  tensor.Vector[T]
	min, max tensor.Vector[T]
}

func Gaussian[T constraints.Float](
	u tensor.Vector[T],
	s tensor.Matrix[T],
	min tensor.Vector[T],
	max tensor.Vector[T],
) Distribution[T] {
	var g gaussian[T]
	g.u = u
	g.s = s
	g.init(min, max)
	return g
}

func (g *gaussian[T]) init(min, max tensor.Vector[T]) {
	g.min = min
	g.max = max

	g.shape = make(tensor.Indices, min.Dim())
	for i := 0; i < min.Dim(); i++ {
		g.shape[i] = 256
	}
	var indices = make(tensor.Indices, g.shape.Len())
	var cur = make(tensor.Vector[T], min.Dim())
	var offset int
	g.weights = make(tensor.Vector[T], tensor.SizeOf(g.shape))
	for len(indices) > 0 {
		for i := range cur {
			var pi = indices.At(i)
			cur[i] = (T(pi) + 0.5) * (max[i] - min[i]) / T(g.shape.At(i))
		}
		var d = cur.Sub(g.u)
		g.weights[offset] = T(math.Exp(-float64(d.Dot(g.s.DotVec(d)) / 2)))
		indices = tensor.Next(g.shape, indices)
		offset++
	}
	g.weights = slices.Map(g.weights, func(x T) T {
		g.total += x
		return g.total
	})
}

func (g gaussian[T]) Generate() model.Sample[T] {
	var indices = tensor.IndexOf(g.shape, random(g.total, g.weights), nil)
	var cur = make(tensor.Vector[T], g.min.Dim())
	for i := range cur {
		var pi = indices.At(i)
		cur[i] = (T(pi) + 0.5) * (g.max[i] - g.min[i]) / T(g.shape.At(i))
	}
	return model.Sample[T]{
		Attributes: cur,
	}
}

type mixtureDistribution[T constraints.Float] struct {
	total         T
	weights       []T
	distributions []Distribution[T]
}

func (m mixtureDistribution[T]) Generate() model.Sample[T] {
	var i = random(m.total, m.weights)
	var s = m.distributions[i].Generate()
	s.Label = T(i)
	return s
}

func MixtureDistribution[T constraints.Float](
	weights []T,
	distributions []Distribution[T],
) Distribution[T] {
	var sum T
	var m = &mixtureDistribution[T]{
		weights: slices.Map(weights, func(w T) T {
			sum += w
			return sum
		}),
		distributions: distributions,
	}
	m.total = slices.Sum(m.weights)
	return m
}

func GenerateClassifierData[
	T constraints.Float,
	D Distribution[T],
](n int, d D) []model.Sample[T] {
	var samples = make([]model.Sample[T], 0, n)
	for i := 0; i < n; i++ {
		samples = append(samples, d.Generate())
	}
	return samples
}
