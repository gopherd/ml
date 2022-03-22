package dtree

import (
	"fmt"

	"github.com/gopherd/brain/model"
	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/container/maps"
	"github.com/gopherd/doge/container/ordered"
	"github.com/gopherd/doge/container/slices"
	"github.com/gopherd/doge/container/tree"
	"github.com/gopherd/doge/math/tensor"
)

// Node represents a node of decision tree
type Node[T any] struct {
	parent   *Node[T]
	children []*Node[T]

	AttributeType  int // attribute for spliting children, valid iff len(children) > 0
	AttributeValue T   // value of attribute
	Label          T   // class of sample
}

// String implements container.Node String method
func (node *Node[T]) String() string {
	if node.parent == nil {
		return "."
	}
	return fmt.Sprintf("attr(%d)=%v:(%v)", node.AttributeType, node.AttributeValue, node.Label)
}

// SetParent sets parent node
func (node *Node[T]) SetParent(parent *Node[T]) {
	node.parent = parent
}

// Parent returns parent node, it implements container.Node Parent method
func (node *Node[T]) Parent() *Node[T] {
	return node.parent
}

// AddChild append a child node
func (node *Node[T]) AddChild(child *Node[T]) {
	child.parent = node
	node.children = append(node.children, child)
}

// NumChild returns number of child, it implements container.Node NumChild method
func (node *Node[T]) NumChild() int {
	return len(node.children)
}

// GetChildByIndex returns i-th child node, it implements container.Node GetChildByIndex method
func (node *Node[T]) GetChildByIndex(i int) *Node[T] {
	return node.children[i]
}

// PolicyFunc used to lookup best attribute for spliting
type PolicyFunc[T constraints.Float] func(samples []model.Sample[T], attrs []int) int

// PruningType represents type of pruning tree
type PruningType int

const (
	NoPruning PruningType = iota
	PrePruning
	PostPruning
)

// Model implements brain.Model
type Model[T constraints.Float] struct {
	policy      PolicyFunc[T]
	pruningType PruningType
	root        *Node[T]
}

func NewModel[T constraints.Float](policy PolicyFunc[T], pruningType PruningType) *Model[T] {
	return &Model[T]{
		policy:      policy,
		pruningType: pruningType,
	}
}

// Stringify format the tree to string
func (m *Model[T]) Stringify(options *tree.Options) string {
	return tree.Stringify[*Node[T]](m.root, options)
}

// Train trains the decision tree
func (m *Model[T]) Train(samples []model.Sample[T]) {
	m.root = new(Node[T])
	if len(samples) == 0 {
		return
	}
	var n = len(samples[0].Attributes)
	var attrs = tensor.RangeN(n)
	var attrValues = make([]*ordered.Map[T, int], len(attrs))
	for i := 0; i < n; i++ {
		attrValues[i] = ordered.NewMap[T, int]()
		for _, x := range samples {
			var k = x.Attributes[i]
			attrValues[i].Insert(k, attrValues[i].Get(k)+1)
		}
	}
	m.generateChildren(m.root, samples, attrValues, attrs)
}

func (m *Model[T]) generateChildren(
	parent *Node[T],
	samples []model.Sample[T],
	attributeValues []*ordered.Map[T, int],
	attributeTypes []int,
) {
	// are all classes same?
	var allSame = true
	for i := range samples {
		if i > 0 && samples[i].Label != samples[i-1].Label {
			allSame = false
			break
		}
	}
	if allSame {
		parent.Label = samples[0].Label
		return
	}

	// are all values same on attrs?
	allSame = true
	for _, attr := range attributeTypes {
		var sameAttr = true
		for j := range samples {
			if j > 0 && samples[j].Attributes[attr] != samples[j-1].Attributes[attr] {
				sameAttr = false
				break
			}
		}
		if !sameAttr {
			allSame = false
			break
		}
	}
	if len(attributeTypes) == 0 || allSame {
		parent.Label = maps.MaxValue(model.Counters(samples)).First
		return
	}

	// lookup best attribute for splitting
	var best = m.policy(samples, attributeTypes)
	var bestAttr = attributeTypes[best]
	var last = len(attributeTypes) - 1
	if best != last {
		attributeTypes[best] = attributeTypes[last]
	}
	attributeTypes = attributeTypes[:last]
	var groups = model.Group(samples, bestAttr)
	var iter = attributeValues[bestAttr].First()
	for iter != nil {
		var attrValue = iter.Key()
		iter = iter.Next()
		var node = new(Node[T])
		node.AttributeType = bestAttr
		node.AttributeValue = attrValue
		parent.AddChild(node)
		if s, ok := groups[attrValue]; ok {
			m.generateChildren(node, s, attributeValues, attributeTypes)
		} else {
			node.Label = maps.MaxValue(model.Counters(samples)).First
		}
	}
}

// postPruning post-pruning decision tree
func (m *Model[T]) postPruning(root *Node[T], samples []model.Sample[T]) {
	panic("TODO")
}

// Predict predicts label for sample
func (m *Model[T]) Predict(x tensor.Vector[T]) T {
	return m.predict(m.root, x)
}

func (m *Model[T]) predict(node *Node[T], x tensor.Vector[T]) T {
	for _, child := range node.children {
		if x[child.AttributeType] == child.AttributeValue {
			return m.predict(child, x)
		}
	}
	return node.Label
}

// RF wraps policy for random forest
func RF[T constraints.Float](policy PolicyFunc[T]) PolicyFunc[T] {
	return func(samples []model.Sample[T], attrs []int) int {
		var n = model.Log2(uint(len(attrs)))
		if n < 1 {
			n = 1
		}
		slices.ShuffleN(attrs, n)
		return policy(samples, attrs[:n])
	}
}
