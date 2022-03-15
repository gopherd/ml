package dtree

import (
	"fmt"

	"github.com/gopherd/brain/stat"
	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/container/maps"
	"github.com/gopherd/doge/container/stringify"
	"github.com/gopherd/doge/math/tensor"
)

// Node represents a node of decision tree
type Node[T any] struct {
	parent   *Node[T]
	children []*Node[T]

	Attribute int // attribute for spliting children, valid iff len(children) > 0
	Value     T   // value of attribute
	Class     int // class of sample
}

// String implements container.Node String method
func (node *Node[T]) String() string {
	if node.parent == nil {
		return "."
	}
	return fmt.Sprintf("%d=>%v:(%d)", node.Attribute, node.Value, node.Class)
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

// Stringify format the tree to string
func Stringify[T any](tree *Node[T], options *stringify.Options) string {
	return stringify.Stringify[*Node[T]](tree, options)
}

// PolicyFunc used to lookup best attribute for spliting
type PolicyFunc[T constraints.Float] func(trainSamples []stat.Sample[T], attrs []int) int

// GenerateTree generates a decision tree
func GenerateTree[T constraints.Float](
	trainSamples []stat.Sample[T],
	policy PolicyFunc[T],
) *Node[T] {
	var root = new(Node[T])
	if len(trainSamples) == 0 {
		return root
	}
	var n = len(trainSamples[0].Attributes)
	var attrs = tensor.RangeN(n)
	var attrValues = make([]map[T]int, len(attrs))
	for i := 0; i < n; i++ {
		attrValues[i] = make(map[T]int)
		for _, x := range trainSamples {
			attrValues[i][x.Attributes[i]]++
		}
	}
	generateChildren(root, trainSamples, attrValues, attrs, policy)
	return root
}

func generateChildren[T constraints.Float](
	parent *Node[T],
	samples []stat.Sample[T],
	attrValues []map[T]int,
	attrs []int,
	policy PolicyFunc[T],
) {
	// are all classes same?
	var allSame = true
	for i := range samples {
		if i > 0 && samples[i].Class != samples[i-1].Class {
			allSame = false
			break
		}
	}
	if allSame {
		parent.Class = samples[0].Class
		return
	}

	// are all values same on attrs?
	allSame = true
	for _, attr := range attrs {
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
	if len(attrs) == 0 || allSame {
		parent.Class = maps.MaxValue(stat.CounterSamples(samples)).First
		return
	}

	// lookup best attribute for splitting
	var best = policy(samples, attrs)
	var bestAttr = attrs[best]
	var last = len(attrs) - 1
	if best != last {
		attrs[best] = attrs[last]
	}
	attrs = attrs[:last]
	var groups = stat.Group(samples, bestAttr)
	for k := range attrValues[bestAttr] {
		var node = new(Node[T])
		node.Attribute = bestAttr
		node.Value = k
		node.SetParent(parent)
		parent.AddChild(node)
		if s, ok := groups[k]; ok {
			generateChildren(node, s, attrValues, attrs, policy)
		} else {
			node.Class = maps.MaxValue(stat.CounterSamples(samples)).First
		}
	}
}
