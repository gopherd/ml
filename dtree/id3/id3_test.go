package id3_test

import (
	"testing"

	"github.com/gopherd/brain/dataloader"
	"github.com/gopherd/brain/dtree"
	"github.com/gopherd/brain/dtree/id3"
)

func TestGenerateTree(t *testing.T) {
	type T = float32
	samples, err := dataloader.LoadCSVFile[T]("../../testdata/watermelon/v2/data.csv")
	if err != nil {
		t.Fatalf("load test data error: %v", err)
	}
	var root = dtree.Generate(samples, id3.Policy[T])
	t.Log(dtree.Stringify(root, nil))
}
