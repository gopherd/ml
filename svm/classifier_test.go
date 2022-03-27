package svm_test

import (
	"math/rand"
	"os"
	"testing"

	"github.com/gopherd/doge/container/slices"
	"github.com/gopherd/doge/math/mathutil"
	"github.com/gopherd/doge/math/tensor"
	"github.com/gopherd/doge/operator"
	"github.com/gopherd/ml/canvas2d"
	"github.com/gopherd/ml/model"
	"github.com/gopherd/ml/svm"
)

func TestSVMClassifier(t *testing.T) {
	type T = float64
	var samples = slices.Map(tensor.RangeN(100), func(i int) model.Sample[T] {
		x := T(rand.Float64())
		y := T(rand.Float64())
		for mathutil.Abs(x-y) < 0.1 {
			y = T(rand.Float64())
		}
		return model.Sample[T]{
			Attributes: tensor.Vec(x, y),
			Label:      operator.If(x < y, 1.0, -1.0),
		}
	})
	var model = svm.NewClassifier[T](1.0, nil)
	var tracker = canvas2d.NewAnimation()
	model.Train(samples, tracker)
	for i := range samples {
		samples[i].Label = model.Predict(samples[i].Attributes)
	}
	tracker.Snapshot(nil)

	if testing.Verbose() {
		file, err := os.Create("svm.gif")
		if err != nil {
			panic(err)
		}
		defer file.Close()
		tracker.Encode(file)
	}
	t.Log(tracker.String())
}
