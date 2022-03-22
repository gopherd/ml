package kmean

import (
	"math"

	"github.com/gopherd/brain/model"
	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/container/pair"
	"github.com/gopherd/doge/container/slices"
	"github.com/gopherd/doge/math/mathutil"
	"github.com/gopherd/doge/math/tensor"
)

func Clustering[T constraints.Float](samples []model.Sample[T], k int) []tensor.Vector[T] {
	if len(samples) <= k {
		var means = make([]tensor.Vector[T], len(samples))
		for i := range samples {
			samples[i].Label = T(i)
			means[i] = samples[i].Attributes
		}
		return means
	}
	var means = slices.Map(slices.ShuffleN(tensor.RangeN(len(samples)), k)[:k], func(i int) tensor.Vector[T] {
		return samples[i].Attributes
	})
	var count = tensor.Repeat(0, k)
	for i := range samples {
		samples[i].Label = T(-1)
	}
	for times := 0; times < len(samples); times++ {
		var updated int
		for i := range samples {
			var x = samples[i].Attributes
			var min pair.Pair[int, T]
			for j := range means {
				var y = means[j]
				var squared T
				for k := range x {
					var d = x[k] - y[k]
					squared += d * d
				}
				if j == 0 || squared < min.Second {
					min.First = j
					min.Second = squared
				}
			}
			var label = T(min.First)
			if label != samples[i].Label {
				samples[i].Label = label
				updated++
			}
		}
		if updated == 0 {
			break
		}
		slices.CopyFunc(count, count, mathutil.Zero[int])
		for i := range samples {
			var x = samples[i].Attributes
			var label = int(samples[i].Label)
			var mean = means[label]
			if count[label] == 0 {
				for j := range mean {
					mean[j] = 0
				}
			}
			count[label]++
			for j := range mean {
				mean[j] += x[j]
			}
		}
		for i := range means {
			if count[i] > 1 {
				for j := range means[i] {
					means[i][j] /= T(count[i])
				}
			}
		}
	}
	return means
}

type box struct {
	items map[int]bool
}

func AutoClustering[T constraints.Float](
	samples []model.Sample[T],
	radius T,
	w model.AffinityFunc[T],
) []tensor.Vector[T] {
	if len(samples) == 0 {
		return nil
	}
	var min, max = model.Minmax(samples)
	var shape = make(tensor.Indices, min.Dim())
	for i := 0; i < shape.Len(); i++ {
		shape[i] = int(math.Ceil(float64((max[i] - min[i]) / radius)))
	}
	var boxes = make([]box, tensor.SizeOf(shape))
	var indices = make(tensor.Indices, shape.Len())
	var mapping = make(map[int]int)

	// lookup index of box for each sample
	for i := range samples {
		x := samples[i].Attributes
		for j := range x {
			v := int(math.Ceil(float64(x[j]-min[j]) / radius))
			if v < 0 {
				v = 0
			} else if v >= shape.At(j) {
				v = shape.At(j) - 1
			}
			indices[j] = v
		}
		var offset = tensor.OffsetOf(shape, indices)
		if boxes[offset].items {
			boxes[offset].items = make(map[int]bool)
		}
		boxes[offset].items[i] = true
		mapping[i] = offset
	}

	for i := range boxes {
	}

	return nil
}
