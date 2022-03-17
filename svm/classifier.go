// implements SVM Classifier
//
package svm

import (
	"github.com/gopherd/brain/stat"
	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/math/tensor"
)

// linear classifier: y = wx + b
type Classifier[T constraints.SignedReal] struct {
	w tensor.Vector[T]
	b T
	k Kernel[T]
}

func NewClassifier[T constraints.SignedReal](k)

func (c *Classifier[T]) Train(samples []stat.Sample[T]) {
}
