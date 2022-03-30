package poker

import (
	"bytes"
	"sort"
	"strconv"
)

type Suit uint8

const (
	Spade   Suit = 0 // "♠"
	Heart   Suit = 1 // "♥"
	Club    Suit = 2 // "♣"
	Diamond Suit = 3 // "♦"
)

type Value uint8

const (
	InvalidPokerValue Value = 0

	PA      Value = 1
	P2      Value = 2
	P3      Value = 3
	P4      Value = 4
	P5      Value = 5
	P6      Value = 6
	P7      Value = 7
	P8      Value = 8
	P9      Value = 9
	P10     Value = 10
	PJ      Value = 11
	PQ      Value = 12
	PK      Value = 13
	PMA     Value = 14
	PM2     Value = 15
	PJoker1 Value = 16
	PJoker2 Value = 17
)

func (v Value) String() string {
	if v >= P2 && v < P10 {
		return strconv.Itoa(int(v))
	}
	switch v {
	case P10:
		return "X"
	case PM2:
		return "2"
	case PA, PMA:
		return "A"
	case PJ:
		return "J"
	case PQ:
		return "Q"
	case PK:
		return "K"
	case PJoker1:
		return "#"
	case PJoker2:
		return "$"
	default:
		return "U"
	}
}

func FormatValues(values []Value) string {
	var head bytes.Buffer
	var body bytes.Buffer
	var foot bytes.Buffer
	head.WriteString("┏")
	body.WriteString("┃")
	foot.WriteString("┗")
	for i := range values {
		if i > 0 {
			head.WriteString("┳")
			body.WriteString("┃")
			foot.WriteString("┻")
		}
		head.WriteString("━")
		body.WriteString(values[i].String())
		foot.WriteString("━")
	}
	head.WriteString("┓")
	body.WriteString("┃")
	foot.WriteString("┛")
	head.WriteByte('\n')
	head.Write(body.Bytes())
	head.WriteByte('\n')
	head.Write(foot.Bytes())
	return head.String()
}

const (
	Joker1 Poker = Poker(Spade<<5) | Poker(PJoker1)
	Joker2 Poker = Poker(Spade<<5) | Poker(PJoker2)
)

// Poker represents a poker with suit and value
type Poker uint8

func Make(suit Suit, value Value) Poker {
	return Poker(int(suit<<5) | int(value))
}

func (p Poker) Suit() Suit {
	return Suit((p >> 5) & 0x3)
}

func (p Poker) Value() Value {
	return Value(p & 0x1F)
}

func (p Poker) Less(p2 Poker) bool {
	v1, v2 := p.Value(), p2.Value()
	return v1 < v2 || (v1 == v2 && p.Suit() > p2.Suit())
}

func (p Poker) Greater(p2 Poker) bool {
	v1, v2 := p.Value(), p2.Value()
	return v1 < v2 || (v1 == v2 && p.Suit() < p2.Suit())
}

func (p Poker) IsJoker1() bool {
	return p.Value() == PJoker1
}

func (p Poker) IsJoker2() bool {
	return p.Value() == PJoker2
}

func (p Poker) IsJoker() bool {
	value := p.Value()
	return value == PJoker1 || value == PJoker2
}

func (p Poker) String() string {
	var suit string
	var value = p.Value()
	if !p.IsJoker() {
		switch p.Suit() {
		case Spade:
			suit = "♠"
		case Heart:
			suit = "♥"
		case Club:
			suit = "♣"
		case Diamond:
			suit = "♦"
		}
	}
	return suit + value.String()
}

// Sort sorts pokers by poker order
func Sort(pokers []Poker) {
	sort.Sort(byPoker(pokers))
}

type byPoker []Poker

func (by byPoker) Len() int           { return len(by) }
func (by byPoker) Less(i, j int) bool { return by[i] < by[j] }
func (by byPoker) Swap(i, j int)      { by[i], by[j] = by[j], by[i] }

// SortByValue sorts pokers by poker value
func SortByValue(pokers []Poker) {
	sort.Sort(byValue(pokers))
}

type byValue []Poker

func (by byValue) Len() int           { return len(by) }
func (by byValue) Less(i, j int) bool { return by[i].Value() < by[j].Value() }
func (by byValue) Swap(i, j int)      { by[i], by[j] = by[j], by[i] }
