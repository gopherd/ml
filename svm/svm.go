package svm

import (
	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/math/tensor"
)

type Kernel[T constraints.SignedReal] func(tensor.Vector[T], tensor.Vector[T]) T

func sign[T constraints.SignedReal](x T) T {
	if x < 0 {
		return -1
	}
	return 1
}
