package canvas2d

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/color/palette"
	"image/gif"
	"image/png"
	"io"
	"sort"

	"github.com/wcharczuk/go-chart/v2"
	"github.com/wcharczuk/go-chart/v2/drawing"

	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/container/maps"
	"github.com/gopherd/doge/container/slices"
	"github.com/gopherd/doge/math/mathutil"
	"github.com/gopherd/doge/mime/mime64"
	"github.com/gopherd/doge/operator"
	"github.com/gopherd/ml/model"
)

type Image = model.Image
type Color = drawing.Color

func NewImage(rect image.Rectangle) *Image {
	return image.NewPaletted(rect, palette.Plan9)
}

func Clone(img *Image) *Image {
	img2 := *img
	img2.Pix = make([]uint8, len(img.Pix))
	copy(img2.Pix, img.Pix)
	return &img2
}

const Size = 256

const XAxis = 0
const YAxis = 1

func Attributes[T constraints.Float](samples []model.Sample[T], attributeType int) []float64 {
	return slices.Map(samples, func(s model.Sample[T]) float64 {
		return float64(s.Attributes[attributeType])
	})
}

func Classes[T constraints.Float](samples []model.Sample[T]) []int {
	return slices.Map(samples, func(s model.Sample[T]) int { return int(s.Label) })
}

func Values[T constraints.Float](values ...T) []float64 {
	return slices.Map(values, func(x T) float64 { return float64(x) })
}

func RGB(c uint32) Color {
	r, g, b := uint8((c>>16)&0xFF), uint8((c>>8)&0xFF), uint8(c&0xFF)
	return Color{R: r, G: g, B: b, A: 0xFF}
}

func RGBA(c uint32) Color {
	r, g, b, a := uint8((c>>24)&0xFF), uint8((c>>16)&0xFF), uint8((c>>8)&0xFF), uint8(c&0xFF)
	return Color{R: r, G: g, B: b, A: a}
}

var defaultColors = []Color{
	RGB(0xB39DDB),
	RGB(0x81D4FA),
	RGB(0x81C784),
	RGB(0xFFE082),
	RGB(0xFFAB91),
	RGB(0xB0BEC5),
	RGB(0xF48FB1),
	RGB(0x7986CB),
	RGB(0x80DEEA),
	RGB(0xC5E1A5),
	RGB(0xBCAAA4),
	RGB(0xCE93D8),
	RGB(0x64B5F6),
	RGB(0xFFCDD2),
	RGB(0x80CBC4),
}

type Canvas struct {
	transformer model.Transformer
	width       int
	height      int
	background  []chart.Series
	figures     []chart.Series
}

func NewCanvas(transformer model.Transformer) *Canvas {
	return &Canvas{
		transformer: transformer,
		width:       Size,
		height:      Size,
	}
}

func (c *Canvas) transform(values []float64, axis int) []float64 {
	return slices.Map(values, func(x float64) float64 {
		return c.transformer.Transform(x, axis)
	})
}

// DrawScatter draws a scatter series to canvas
func (c *Canvas) DrawScatter(x, y []float64, classes []int, options *ScatterOptions) {
	x, y = c.transform(x, XAxis), c.transform(y, YAxis)
	var labels = make(map[int]bool)
	for i := range classes {
		labels[classes[i]] = true
	}
	var orderedLabels = maps.Keys(labels)
	sort.Slice(orderedLabels, func(i, j int) bool {
		return orderedLabels[i] < orderedLabels[j]
	})
	var mapping = make(map[int]int)
	for i, label := range orderedLabels {
		mapping[label] = i
	}
	println(fmt.Sprintf("classes=%v, labels=%v, mapping=%v", classes, labels, mapping))
	if options == nil {
		options = &ScatterOptions{}
	}

	fig := chart.ContinuousSeries{
		Style: chart.Style{
			StrokeWidth: chart.Disabled,
			DotWidth:    operator.If(options.DotSize < 1, 3, options.DotSize),
			DotColorProvider: func(_, _ chart.Range, i int, _, _ float64) Color {
				class := classes[i]
				colors := defaultColors
				if len(options.Colors) >= len(mapping) {
					colors = options.Colors
				}
				index, ok := mapping[class]
				if !ok {
					println("class", class, "not found")
					index = len(colors) - 1
				}
				return colors[index%len(colors)]
			},
		},
		XValues: x,
		YValues: y,
	}
	if options != nil && options.Background {
		c.background = append(c.background, fig)
	} else {
		c.figures = append(c.figures, fig)
	}
}

type ScatterOptions struct {
	Background bool
	Colors     []Color
	DotSize    float64
}

// DrawSegment draws a segment series to canvas
func (c *Canvas) DrawSegment(x, y []float64, options *SegmentOptions) {
	if options == nil {
		options = &SegmentOptions{}
	}
	x, y = c.transform(x, XAxis), c.transform(y, YAxis)
	fig := chart.ContinuousSeries{
		Style: chart.Style{
			StrokeWidth: operator.If(options.LineWidth < 1, 3, options.LineWidth),
			StrokeColor: operator.If(options.Color.IsZero(), drawing.ColorBlack, options.Color),
		},
		XValues: x,
		YValues: y,
	}
	if options != nil && options.Background {
		c.background = append(c.background, fig)
	} else {
		c.figures = append(c.figures, fig)
	}
}

type SegmentOptions struct {
	Background bool
	Color      Color
	LineWidth  float64
}

// Flush flushes all staged graphs to image
func (c *Canvas) Flush() (*Image, error) {
	buf := bytes.NewBuffer(nil)
	graph := chart.Chart{
		Width:  c.width,
		Height: c.height,
		Series: append(c.background, c.figures...),
	}
	c.figures = c.figures[:0]
	if err := graph.Render(chart.PNG, buf); err != nil {
		return nil, err
	}
	img, err := png.Decode(buf)
	if err != nil {
		return nil, err
	}
	p := NewImage(img.Bounds())
	for x := 0; x < p.Rect.Dx(); x++ {
		for y := 0; y < p.Rect.Dy(); y++ {
			p.Set(x, y, img.At(x, y))
		}
	}
	return p, nil
}

type Animation struct {
	canvas *Canvas
	images []*Image
	delay  []int
}

func NewAnimation() *Animation {
	return &Animation{}
}

func (a *Animation) Snapshot(img *Image) {
	if img == nil {
		if len(a.images) > 0 {
			a.delay = append(a.delay, 100)
			a.images = append(a.images, a.images[len(a.images)-1])
		}
		return
	}
	a.delay = append(a.delay, operator.If(len(a.images) == 0, 0, 10))
	a.images = append(a.images, img)
}

func (a *Animation) Encode(w io.Writer) error {
	if len(a.images) == 0 {
		return nil
	}
	b := a.images[0].Bounds()
	if b.Dx() >= 1<<16 || b.Dy() >= 1<<16 {
		return errors.New("gif: image is too large to encode")
	}

	m := &gif.GIF{
		Image: a.images,
		Delay: a.delay,
		Config: image.Config{
			ColorModel: a.images[0].ColorModel(),
			Width:      b.Dx(),
			Height:     b.Dy(),
		},
	}

	return gif.EncodeAll(w, m)
}

func (a *Animation) String() string {
	var buf bytes.Buffer
	if err := a.Encode(&buf); err != nil {
		return err.Error()
	}
	return mime64.EncodeToString(buf.Bytes())
}

func ClipSegment[T constraints.Float](a, b, c, xmin, xmax, ymin, ymax T) (x0, y0, x1, y1 T, ok bool) {
	if mathutil.Abs(a) < mathutil.Abs(b) {
		x0, y0, x1, y1, ok = clipSegment(a, b, c, xmin, xmax, ymin, ymax)
	} else {
		y0, x0, y1, x1, ok = clipSegment(b, a, c, ymin, ymax, xmin, xmax)
	}
	return
}

func clipSegment[T constraints.Float](a, b, c, xmin, xmax, ymin, ymax T) (x0, y0, x1, y1 T, ok bool) {
	a /= b
	c /= b
	b = 1
	x0, x1 = xmin, xmax
	y0, y1 = -a*x0-c, -a*x1-c
	if a < -model.Epsilon || a > model.Epsilon {
		if y0 < ymin {
			x0 = -(ymin + c) / a
		} else if y0 > ymax {
			x0 = -(ymax + c) / a
		}
		if y1 < ymin {
			x1 = -(ymin + c) / a
		} else if y1 > ymax {
			x1 = -(ymax + c) / a
		}
		y0, y1 = -a*x0-c, -a*x1-c
	}
	ok = (x0 <= xmax || x1 <= xmax) && (x0 >= xmin || x1 >= xmin) &&
		(y0 <= ymax || y1 <= ymax) && (y0 >= ymin || y1 >= ymin)
	return
}
