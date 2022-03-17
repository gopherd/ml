package dtree_test

import (
	"testing"

	"github.com/gopherd/brain/dataloader"
	"github.com/gopherd/brain/dtree"
	"github.com/gopherd/brain/stat"
)

func TestGenerateTree(t *testing.T) {
	type T = float32
	samples, err := dataloader.LoadCSVFile[T]("../testdata/watermelon/v2/data.csv")
	if err != nil {
		t.Fatalf("load test data error: %v", err)
	}
	var root = dtree.Generate(samples, func(trainSamples []stat.Sample[T], attrs []int) int {
		return len(attrs) - 1
	})
	t.Log(dtree.Stringify(root, nil))
}
