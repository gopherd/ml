package kuhn_test

import (
	"testing"

	"github.com/gopherd/ml/cfr/kuhn"
)

func TestKuhnTrain(t *testing.T) {
	type T = float64
	const iterations = 100000
	var k = kuhn.NewKuhnPoker[T](iterations)
	var value = k.Train()
	t.Logf("value: %v", value)
}
