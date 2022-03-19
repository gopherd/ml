package model

import (
	"github.com/gopherd/brain/stat"
	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/math/tensor"
)

type Model[T constraints.Float] interface {
	Train(samples []stat.Sample[T])
	Predict(x tensor.Vector[T]) T
}
