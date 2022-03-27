package id3_test

import (
	"testing"

	"github.com/gopherd/ml/dtree"
	"github.com/gopherd/ml/dtree/id3"
)

func TestModel(t *testing.T) {
	type T = float32
	var model = dtree.NewModel(id3.Policy[T], dtree.NoPruning)
	dtree.TestModel("../../testdata/watermelon/v2/data.csv", model, t)
	t.Logf("\n%v", model.Stringify(nil))
}
