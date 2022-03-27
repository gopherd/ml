package canvas2d_test

import (
	"testing"

	"github.com/gopherd/ml/canvas2d"
)

func TestClipSegment(t *testing.T) {
	type T float64
	const size = 16
	var xmin, xmax T = 0, size
	var ymin, ymax T = 0, size
	type testCase struct {
		a, b, c        T
		x0, y0, x1, y1 T
		ok             bool
	}
	for i, tc := range []testCase{
		{
			1, -1, 0,
			0, 0, size, size,
			true,
		},
		{
			1, 0, -1,
			1, 0, 1, size,
			true,
		},
		{
			0, 1, -1,
			0, 1, size, 1,
			true,
		},
		{
			1, 2, -2, // x+2y-2=0
			0, 1, 2, 0,
			true,
		},
		{
			1, 2, -size * 4, // x+2y-64=0
			0, 0, 0, 0,
			false,
		},
	} {
		x0, y0, x1, y1, ok := canvas2d.ClipSegment(tc.a, tc.b, tc.c, xmin, xmax, ymin, ymax)
		if tc.ok != ok {
			t.Fatalf("%dth: ClipSegment: want %v, got %v", i, tc.ok, ok)
		}
		if ok {
			if tc.x0 != x0 || tc.y0 != y0 || tc.x1 != x1 || tc.y1 != y1 {
				t.Fatalf("%dth: ClipSegment: want (%v,%v,%v,%v), got (%v,%v,%v,%v)",
					i, tc.x0, tc.y0, tc.x1, tc.y1, x0, y0, x1, y1,
				)
			}
		}
	}
}
