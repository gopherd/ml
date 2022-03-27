package kmeans_test

import (
	"math/rand"
	"testing"

	"github.com/gopherd/doge/math/tensor"
	"github.com/gopherd/ml/kmeans"
	"github.com/gopherd/ml/model"
)

func TestKMean(t *testing.T) {
	type T = float64
	var samples = make([]model.Sample[T], 1<<14)
	const k = 4
	const interval = 1
	for i := range samples {
		label := T(i % k)
		x := label*interval + (rand.Float64()*0.5-0.25)*interval
		samples[i].Attributes = tensor.Vec(x)
	}
	var means = kmeans.Clustering(samples, k)
	t.Logf("means: %v", means)
}
