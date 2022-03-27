// implements SVM Classifier
//
package svm

import (
	"math/rand"

	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/container/pair"
	"github.com/gopherd/doge/math/mathutil"
	"github.com/gopherd/doge/math/tensor"
	"github.com/gopherd/doge/operator"
	"github.com/gopherd/ml/canvas2d"
	"github.com/gopherd/ml/model"
)

// linear classifier: f(x) = Σᵢ(aᵢ‧k(x,xᵢ)) + b
type Classifier[T constraints.Float] struct {
	// len(a) == len(s), s=[(x,y)]
	a tensor.Vector[T]
	s []model.Sample[T]
	b T
	k Kernel[T]
	c T

	min, max tensor.Vector[T]
}

func NewClassifier[T constraints.Float](c T, kernel Kernel[T]) *Classifier[T] {
	return &Classifier[T]{
		k: kernel,
		c: c,
	}
}

func (c *Classifier[T]) kernel(x, y tensor.Vector[T]) T {
	if c.k == nil {
		return x.Dot(y)
	}
	return c.k(x, y)
}

func (c *Classifier[T]) Snapshot() *canvas2d.Image {
	if c.k != nil || len(c.s) == 0 || c.s[0].Attributes.Dim() != 2 {
		return nil
	}
	canvas := canvas2d.NewCanvas(model.NewTransformer(canvas2d.Size, c.min, c.max))
	// draw scatter
	canvas.DrawScatter(
		canvas2d.Attributes(c.s, 0),
		canvas2d.Attributes(c.s, 1),
		canvas2d.Classes(c.s),
		nil,
	)
	if len(c.a) > 0 {
		// draw line: ax + by + c = 0
		var a, b T
		for i := range c.a {
			a += c.a[i] * c.s[i].Attributes[0]
			b += c.a[i] * c.s[i].Attributes[1]
		}
		x0, y0, x1, y1, ok := canvas2d.ClipSegment(a, b, c.b, c.min[0], c.max[0], c.min[1], c.max[1])
		if ok {
			canvas.DrawSegment(canvas2d.Values(x0, x1), canvas2d.Values(y0, y1), nil)
		}
	}
	img, err := canvas.Flush()
	if err != nil {
		return nil
	}
	return img
}

func (c *Classifier[T]) Train(samples []model.Sample[T], tracker model.Tracker) {
	c.min, c.max = model.Minmax(samples)
	c.s = samples
	c.a = make([]T, len(c.s))
	for i := range c.a {
		c.a[i] = T(rand.Float64()) * c.c
	}

	if tracker != nil {
		tracker.Snapshot(c.Snapshot())
	}

	c.smo(samples, tracker)

	if tracker != nil {
		tracker.Snapshot(c.Snapshot())
	}
}

// implements SMO algorithm
func (c *Classifier[T]) smo(samples []model.Sample[T], tracker model.Tracker) {
	const maxIdle = 16
	var idle int
	for idle < maxIdle {
		idle++
		// step1: select index pair (pi, pj)
		pi, pj := c.selectAlpha()
		if pi.First < 0 {
			break
		}
		i, j := pi.First, pj.First

		// step2: update a[i], a[j]
		ai, aj := c.a[i], c.a[j]
		xi, xj := c.s[i].Attributes, c.s[j].Attributes
		yi, yj := sign(c.s[i].Label), sign(c.s[j].Label)
		ei, ej := mathutil.Abs(yi-pi.Second), mathutil.Abs(yj-pj.Second)
		kii, kjj, kij := c.kernel(xi, xi), c.kernel(xj, xj), c.kernel(xi, xj)
		eta := kii + kjj - 2*kij
		c.a[j] = c.a[j] + yj*(ei-ej)/eta
		if c.c > 0 {
			u := operator.If(yi*yj < 0, mathutil.Max(0, aj-ai), mathutil.Max(0, ai+aj-c.c))
			v := operator.If(yi*yj < 0, mathutil.Min(c.c, aj-ai+c.c), mathutil.Min(c.c, ai+aj))
			c.a[j] = mathutil.Clamp(c.a[j], u, v)
		}
		c.a[i] += yi * yj * (aj - c.a[j])
		if mathutil.Abs(c.a[i]-ai) < model.Epsilon && mathutil.Abs(c.a[j]-aj) < model.Epsilon {
			continue
		}

		// step3: update bias
		var n int
		var b T
		var bi = -ei + ai*yi*kii + aj*yj*kij + c.b
		var bj = -ej - (c.a[i]-ai)*yi*kij - (c.a[j]-aj)*yj*kjj + c.b
		if c.a[i] > model.Epsilon && c.a[i] < c.c-model.Epsilon {
			n++
			b += bi
		}
		if c.a[j] > model.Epsilon && c.a[j] < c.c-model.Epsilon {
			n++
			b += bj
		}
		if n == 0 {
			n = 2
			b = bi + bj
		}
		c.b = b / T(n)
		idle = 0
		if tracker != nil {
			tracker.Snapshot(c.Snapshot())
		}
	}

	// remove zeros from a and save relative samples(support vectors)
	var n int
	for i := range c.a {
		if c.a[i] > 0 {
			n++
		}
	}
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

	c.updateBias()
}

func (c *Classifier[T]) selectAlpha() (max, far pair.Pair[int, T]) {
	if len(c.a) < 2 {
		max.First = -1
		return
	}
	max.First = rand.Intn(len(c.a))
	max.Second = c.Predict(c.s[max.First].Attributes)
	var farDist T
	far.First = -1
	for i := range c.a {
		if i == max.First {
			continue
		}
		var dist T
		for j, x := range c.s[i].Attributes {
			dx := c.s[max.First].Attributes[j] - x
			dist += dx * dx
		}
		if far.First < 0 || dist > farDist {
			far.First = i
			farDist = dist
		}
	}
	far.Second = c.Predict(c.s[far.First].Attributes)
	return max, far
}

func (c *Classifier[T]) updateBias() {
	if len(c.s) == 0 {
		return
	}
	var sum T
	for i := range c.s {
		sum += sign(c.s[i].Label)
		for j := range c.s {
			sum -= c.a[j] * sign(c.s[j].Label) * c.kernel(c.s[i].Attributes, c.s[j].Attributes)
		}
	}
	c.b = sum / T(len(c.s))
}

func (c *Classifier[T]) Predict(x tensor.Vector[T]) T {
	var sum = c.b
	for i := range c.a {
		sum += c.a[i] * c.kernel(x, c.s[i].Attributes)
	}
	return sign(sum)
}
