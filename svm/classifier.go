// implements SVM Classifier
//
package svm

import (
	"github.com/gopherd/brain/stat"
	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/math/tensor"
	"github.com/gopherd/doge/operator"
)

// linear classifier: f(x) = Σᵢ(aᵢ‧k(x,xᵢ)) + b
type Classifier[T constraints.Float] struct {
	// len(a) == len(s), s=[(x,y)]
	a tensor.Vector[T]
	s []stat.Sample[T]
	b T
	k Kernel[T]
}

func NewClassifier[T constraints.Float](kernel Kernel[T]) *Classifier[T] {
	return &Classifier[T]{
		k: operator.If(kernel == nil, dotv[T], kernel),
	}
}

func (c *Classifier[T]) Train(samples []stat.Sample[T]) {
	// TODO:implements SMO algorithm
}

func (c *Classifier[T]) Predict(x tensor.Vector[T]) T {
	var sum = c.b
	for i := range c.a {
		sum += c.a[i] * c.k(x, c.s[i].Attributes)
	}
	return sign(sum)
}
