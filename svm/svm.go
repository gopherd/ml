package svm

import (
	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/math/tensor"
)

type Kernel[T constraints.Float] func(tensor.Vector[T], tensor.Vector[T]) T

func sign[T constraints.Float](x T) T {
	if x < 0 {
		return -1
	}
	return 1
}

func dotv[T constraints.Float](a, b tensor.Vector[T]) T {
	return a.Dot(b)
}
