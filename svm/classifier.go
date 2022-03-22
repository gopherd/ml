// implements SVM Classifier
//
package svm

import (
	"math/rand"

	"github.com/gopherd/brain/model"
	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/container/pair"
	"github.com/gopherd/doge/math/mathutil"
	"github.com/gopherd/doge/math/tensor"
	"github.com/gopherd/doge/operator"
)

// linear classifier: f(x) = Σᵢ(aᵢ‧k(x,xᵢ)) + b
type Classifier[T constraints.Float] struct {
	// len(a) == len(s), s=[(x,y)]
	a tensor.Vector[T]
	s []model.Sample[T]
	b T
	k Kernel[T]
	c T
}

func NewClassifier[T constraints.Float](kernel Kernel[T]) *Classifier[T] {
	return &Classifier[T]{
		k: operator.If(kernel == nil, dotv[T], kernel),
	}
}

func (c *Classifier[T]) Train(samples []model.Sample[T]) {
	c.smo(samples)
}

// implements SMO algorithm
func (c *Classifier[T]) smo(samples []model.Sample[T]) {
	// initialize
	c.s = samples
	c.a = make([]T, len(c.s))
	for i := range c.a {
		c.a[i] = T(rand.Float64()) * c.c
	}
	var supports = tensor.RangeN[int](len(c.a))
	c.updateBias(supports)

	for {
		// step1: select index pair (pi, pj)
		pi, pj := c.selectAlpha()
		if pi.First < 0 {
			break
		}
		i, j := pi.First, pj.First

		// step2: update a[i], a[j]
		ai, aj := c.a[i], c.a[j]
		xi, xj := c.s[i].Attributes, c.s[j].Attributes
		yi, yj := c.s[i].Label, c.s[j].Label
		ei, ej := mathutil.Abs(yi-pi.Second), mathutil.Abs(yj-pj.Second)
		kii, kjj, kij := c.k(xi, xi), c.k(xj, xj), c.k(xi, xj)
		ksum := kii + kjj - 2*kij
		u := operator.If(yi*yj < 0, mathutil.Max(0, aj-ai), mathutil.Max(0, ai+aj-c.c))
		v := operator.If(yi*yj < 0, mathutil.Min(c.c, aj-ai+c.c), mathutil.Min(c.c, ai+aj))
		c.a[j] = mathutil.Clamp(c.a[j]+yj*(ei-ej)/ksum, u, v)
		c.a[i] += yi * yj * (aj - c.a[j])

		// step3: update bias
		c.updateBias(supports)
	}

	// remove zeros from a and save relative samples(support vectors)
	var n int
	for i := range c.a {
		if c.a[i] > 0 {
			n++
		}
	}
	c.s = make([]model.Sample[T], n)
	n = 0
	for i := range c.a {
		if c.a[i] > 0 {
			c.a[n] = c.a[i]
			c.s[n] = c.s[i]
			n++
		}
	}
	c.a = c.a[:n]
	c.s = c.s[:n]
}

func (c *Classifier[T]) selectAlpha() (max, far pair.Pair[int, T]) {
	var maxDiff T
	for i := range c.a {
		var diff T
		var y = c.Predict(c.s[i].Attributes)
		if c.a[i] == 0 {
			if y < 1 {
				diff = 1 - y
			}
		} else if c.a[i] == c.c {
			if y > 1 {
				diff = y - 1
			}
		} else {
			if y != 1 {
				diff = mathutil.Abs(y - 1)
			}
		}
		if i == 0 || diff > maxDiff {
			max.First = i
			max.Second = y
			maxDiff = diff
		}
	}
	if max.Second == 0 {
		max.First = -1
		return
	}
	var farDiff T
	for i := range c.a {
		var y = c.Predict(c.s[i].Attributes)
		var diff = mathutil.Abs(y - max.Second)
		if i == 0 || diff > farDiff {
			far.First = i
			far.Second = y
			farDiff = diff
		}
	}
	return max, far
}

func (c *Classifier[T]) updateBias(supports []int) {
	if len(supports) == 0 {
		return
	}
	var sum T
	for _, i := range supports {
		sum += c.s[i].Label
		for _, j := range supports {
			sum -= c.a[j] * c.s[j].Label * c.k(c.s[i].Attributes, c.s[j].Attributes)
		}
	}
	c.b = sum / T(len(supports))
}

func (c *Classifier[T]) Predict(x tensor.Vector[T]) T {
	var sum = c.b
	for i := range c.a {
		sum += c.a[i] * c.k(x, c.s[i].Attributes)
	}
	return sign(sum)
}
