package dtree

import (
	"fmt"

	"github.com/gopherd/brain/stat"
	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/container/maps"
	"github.com/gopherd/doge/container/ordered"
	"github.com/gopherd/doge/container/stringify"
	"github.com/gopherd/doge/math/tensor"
)

// Node represents a node of decision tree
type Node[T any] struct {
	parent   *Node[T]
	children []*Node[T]

	AttributeType  int // attribute for spliting children, valid iff len(children) > 0
	AttributeValue T   // value of attribute
	Class          int // class of sample
}

// String implements container.Node String method
func (node *Node[T]) String() string {
	if node.parent == nil {
		return "."
	}
	return fmt.Sprintf("attr(%d)=%v:(%d)", node.AttributeType, node.AttributeValue, node.Class)
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

// Stringify format the tree to string
func Stringify[T any](tree *Node[T], options *stringify.Options) string {
	return stringify.Stringify[*Node[T]](tree, options)
}

// PolicyFunc used to lookup best attribute for spliting
type PolicyFunc[T constraints.Float] func(trainSamples []stat.Sample[T], attrs []int) int

// Generate generates a decision tree
func Generate[T constraints.Float](
	trainSamples []stat.Sample[T],
	policy PolicyFunc[T],
) *Node[T] {
	var root = new(Node[T])
	if len(trainSamples) == 0 {
		return root
	}
	var n = len(trainSamples[0].Attributes)
	var attrs = tensor.RangeN(n)
	var attrValues = make([]*ordered.Map[T, int], len(attrs))
	for i := 0; i < n; i++ {
		attrValues[i] = ordered.NewMap[T, int]()
		for _, x := range trainSamples {
			var k = x.Attributes[i]
			attrValues[i].Insert(k, attrValues[i].Get(k)+1)
		}
	}
	generateChildren(root, trainSamples, attrValues, attrs, policy)
	return root
}

func generateChildren[T constraints.Float](
	parent *Node[T],
	samples []stat.Sample[T],
	attributeValues []*ordered.Map[T, int],
	attributeTypes []int,
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
		parent.Class = maps.MaxValue(stat.Counters(samples)).First
		return
	}

	// lookup best attribute for splitting
	var best = policy(samples, attributeTypes)
	var bestAttr = attributeTypes[best]
	var last = len(attributeTypes) - 1
	if best != last {
		attributeTypes[best] = attributeTypes[last]
	}
	attributeTypes = attributeTypes[:last]
	var groups = stat.Group(samples, bestAttr)
	var iter = attributeValues[bestAttr].First()
	for iter != nil {
		var attrValue = iter.Key()
		iter = iter.Next()
		var node = new(Node[T])
		node.AttributeType = bestAttr
		node.AttributeValue = attrValue
		parent.AddChild(node)
		if s, ok := groups[attrValue]; ok {
			generateChildren(node, s, attributeValues, attributeTypes, policy)
		} else {
			node.Class = maps.MaxValue(stat.Counters(samples)).First
		}
	}
}

// TODO: PostPruning post-pruning decision tree
func PostPruning[
	S ~[]stat.Sample[T],
	T constraints.Float,
](root *Node[T], testSamples S) {
}
