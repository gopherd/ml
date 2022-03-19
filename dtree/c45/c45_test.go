package c45_test

import (
	"testing"

	"github.com/gopherd/brain/dtree"
	"github.com/gopherd/brain/dtree/c45"
)

func TestModel(t *testing.T) {
	type T = float32
	var model = dtree.NewModel(c45.Policy[T], dtree.NoPruning)
	dtree.TestModel("../../testdata/watermelon/v2/data.csv", model, t)
	t.Log(model.Stringify(nil))
}
