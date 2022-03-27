package model

import (
	"image"

	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/math/tensor"
)

type Image = image.Paletted

type Tracker interface {
	Snapshot(*Image)
}

type Transformer interface {
	Transform(x float64, i int) float64
}

type transformer struct {
	scale     tensor.Vector[float64]
	translate tensor.Vector[float64]
}

func (t transformer) Transform(x float64, i int) float64 {
	return x*t.scale[i] + t.translate[i]
}

func NewTransformer[T constraints.Float](size T, min, max tensor.Vector[T]) Transformer {
	const padding = 0.1 // 10%
	var dx = max[0] - min[0]
	var dy = max[1] - min[1]
	var t transformer
	t.scale = make(tensor.Vector[float64], 2)
	t.translate = make(tensor.Vector[float64], 2)
	if dx < dy {
		t.scale[0] = float64(size * (1 - 2*padding) / dy)
	} else {
		t.scale[0] = float64(size * (1 - 2*padding) / dx)
	}
	t.scale[1] = t.scale[0]
	dx *= T(t.scale[0])
	dy *= T(t.scale[1])
	t.translate[0] = float64(size-dx) / 2
	t.translate[1] = float64(size-dy) / 2
	return t
}
