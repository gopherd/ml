package c45_test

import (
	"testing"

	"github.com/gopherd/ml/dtree"
	"github.com/gopherd/ml/dtree/c45"
)

func TestModel(t *testing.T) {
	type T = float32
	var model = dtree.NewModel(c45.Policy[T], dtree.NoPruning)
	dtree.TestModel("../../testdata/watermelon/v2/data.csv", model, t)
	t.Logf("\n%v", model.Stringify(nil))
}
