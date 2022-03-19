package dtree

import (
	"math/rand"
	"testing"
	"time"

	"github.com/gopherd/brain/dataloader"
	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/container/slices"
)

// TestModel tests the model with train data from file
func TestModel[T constraints.Float](filename string, m *Model[T], t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	samples, err := dataloader.LoadCSVFile[T](filename)
	if err != nil {
		t.Fatalf("load test data error: %v", err)
	}
	slices.Shuffle(samples)
	var split = len(samples) * 4 / 5
	var trainData = samples[:split]
	var testData = samples[split:]
	m.Train(trainData)
	var accurracy T
	for _, x := range testData {
		label := m.Predict(x.Attributes)
		if x.Label == label {
			accurracy++
		}
	}
	accurracy /= T(len(testData))
	accurracy *= 100
	t.Logf("accurray: %d.%d%%", int(accurracy), int(accurracy*10)/100)
}
