// package kuhn solves the Kuhn poker game.
//
// @see https://en.wikipedia.org/wiki/Kuhn_poker
//
package kuhn

import (
	"bytes"
	"fmt"
	"math/rand"
	"strconv"
	"strings"

	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/container/slices"
	"github.com/gopherd/doge/math/mathutil"
	"github.com/gopherd/doge/operator"
)

const (
	Pass       = 0
	Bet        = 1
	NumActions = 2
)

var actions = []string{
	"p", // pass(fold)
	"b", // bet(call)
}

type Node[T constraints.Float] struct {
	infoSet     string
	regretSum   [NumActions]T
	strategy    [NumActions]T
	strategySum [NumActions]T
}

func (node *Node[T]) updateStrategy(p T) {
	var sum T
	for i := 0; i < NumActions; i++ {
		node.strategy[i] = mathutil.Max(node.regretSum[i], 0)
		sum += node.strategy[i]
	}
	for i := 0; i < NumActions; i++ {
		if sum > 0 {
			node.strategy[i] /= sum
		} else if i == 0 {
			node.strategy[i] = 1.0 / NumActions
		} else {
			node.strategy[i] = node.strategy[i-1]
		}
		node.strategySum[i] += node.strategy[i] * p
	}
}

func (node *Node[T]) getAction() int {
	return node.getActionForStrategy(node.strategySum)
}

func (node *Node[T]) getActionForStrategy(strategy [NumActions]T) int {
	var sum T
	for _, p := range strategy {
		sum += p
	}
	if sum == 0 {
		return rand.Intn(len(strategy))
	}
	var x = T(rand.Float64()) * sum
	for i, p := range strategy {
		sum += p
		if sum > x {
			return i
		}
	}
	return len(strategy) - 1
}

// KuhnPoker implements kuhn poker game learning algorithm
type KuhnPoker[T constraints.Float] struct {
	nodes      map[string]*Node[T]
	iterations int
}

func NewKuhnPoker[T constraints.Float](iterations int) *KuhnPoker[T] {
	return &KuhnPoker[T]{
		nodes:      make(map[string]*Node[T]),
		iterations: iterations,
	}
}

func (k *KuhnPoker[T]) String() string {
	var buf bytes.Buffer
	var n int
	buf.WriteString("{nodes=map[")
	for k, v := range k.nodes {
		if n > 0 {
			buf.WriteByte(' ')
		}
		fmt.Fprint(&buf, k)
		buf.WriteByte(':')
		fmt.Fprint(&buf, v)
		n++
	}
	buf.WriteString("}")
	return buf.String()
}

func (k *KuhnPoker[T]) Train() T {
	var cards = []int{1, 2, 3}
	var util T
	for t := 0; t < k.iterations; t++ {
		slices.Shuffle(cards)
		util += k.cfr(cards, "", 1, 1)
	}
	return util / T(k.iterations)
}

// cfr recursively computes CFR(Counterfactual Regret) value
func (k *KuhnPoker[T]) cfr(cards []int, h string, p0, p1 T) T {
	var plays = len(h)
	var player = plays % 2
	var opponent = 1 - player

	// compute Counterfactual Regret value for terminal state
	if plays > 1 {
		if h[plays-1] == h[plays-2] {
			// cases: bb,pbb,pp
			var payoff = operator.If(h[plays-1] == 'p', T(1.0), T(2.0))
			return operator.If(cards[player] < cards[opponent], -payoff, payoff)
		} else if strings.HasSuffix(h, "p") {
			// cases: pbp,bp
			return 1
		}
	}

	// update local strategy and accumulate average strategy
	var infoSet = strconv.Itoa(cards[player]) + h
	var node, found = k.nodes[infoSet]
	if !found {
		node = new(Node[T])
		node.infoSet = infoSet
		k.nodes[infoSet] = node
	}
	node.updateStrategy(operator.If(player == 0, p0, p1))

	// recursively compute CFR
	var util T
	var utilities [NumActions]T
	for i := 0; i < NumActions; i++ {
		if p := node.strategy[i]; p > 0 {
			if player == 0 {
				utilities[i] = -k.cfr(cards, h+actions[i], p0*p, p1)
			} else {
				utilities[i] = -k.cfr(cards, h+actions[i], p0, p1*p)
			}
			util += utilities[i] * p
		}
	}

	// accumulate regrets
	var rp = operator.If(player == 0, p1, p0)
	for i := 0; i < NumActions; i++ {
		node.regretSum[i] += rp * (utilities[i] - util)
	}

	return util
}
