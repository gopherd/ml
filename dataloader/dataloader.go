package dataloader

import (
	"encoding/csv"
	"errors"
	"io"
	"os"
	"strconv"
	"strings"
	"unsafe"

	"github.com/gopherd/brain/stat"
	"github.com/gopherd/doge/constraints"
	"github.com/gopherd/doge/math/mathutil"
	"github.com/gopherd/doge/math/tensor"
	"github.com/gopherd/doge/operator"
)

type csvOptions struct {
	trimRowHeader    bool
	trimColumnHeader bool
	rows, columns    int
	nolabel          bool
}

func defaultCSVOptions() csvOptions {
	return csvOptions{
		trimRowHeader:    true,
		trimColumnHeader: false,
	}
}

type CSVOption func(opt *csvOptions)

func (opt *csvOptions) apply(options []CSVOption) {
	for _, o := range options {
		o(opt)
	}
}

func WithCSVRowHeader(yes bool) CSVOption {
	return func(opt *csvOptions) {
		opt.trimRowHeader = yes
	}
}

func WithCSVColumnHeader(yes bool) CSVOption {
	return func(opt *csvOptions) {
		opt.trimColumnHeader = yes
	}
}

func WithCSVRows(rows int) CSVOption {
	return func(opt *csvOptions) {
		opt.rows = rows
	}
}

func WithCSVColumns(columns int) CSVOption {
	return func(opt *csvOptions) {
		opt.columns = columns
	}
}

func WithCSVNoLabel(nolabel bool) CSVOption {
	return func(opt *csvOptions) {
		opt.nolabel = nolabel
	}
}

func LoadCSVFile[T constraints.Float](filename string, options ...CSVOption) ([]stat.Sample[T], error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return LoadCSV[T](file, options...)
}

func LoadCSV[T constraints.Float](r io.Reader, options ...CSVOption) ([]stat.Sample[T], error) {
	opt := defaultCSVOptions()
	opt.apply(options)

	reader := csv.NewReader(r)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, nil
	}
	var bits = int(unsafe.Sizeof(T(0))) * 8
	var samples = make([]stat.Sample[T], 0, operator.If(opt.rows > 0, opt.rows, len(records)))
	for i, record := range records {
		if opt.columns < 1 {
			opt.columns = len(record) - mathutil.Predict[int](!opt.nolabel)
			if opt.columns < 1 {
				return nil, errors.New("columns must be greater than 0")
			}
		}
		if opt.rows > 0 && len(samples) == opt.rows {
			break
		}
		if opt.trimRowHeader && i == 0 {
			continue
		}
		var sample stat.Sample[T]
		sample.Attributes = make(tensor.Vector[T], 0, opt.columns)
		for j, s := range record {
			if len(sample.Attributes) == opt.columns {
				if !opt.nolabel {
					label, err := strconv.ParseFloat(s, bits)
					if err != nil {
						return nil, err
					}
					sample.Label = T(label)
				}
				break
			}
			if opt.trimColumnHeader && j == 0 {
				continue
			}
			s = strings.TrimSpace(s)
			if len(s) == 0 {
				sample.Attributes = append(sample.Attributes, 0)
				continue
			}
			value, err := strconv.ParseFloat(s, bits)
			if err != nil {
				return nil, err
			}
			sample.Attributes = append(sample.Attributes, T(value))
		}
		if len(sample.Attributes) < opt.columns {
			if cap(sample.Attributes) >= opt.columns {
				sample.Attributes = sample.Attributes[:opt.columns]
			} else {
				sample.Attributes = append(sample.Attributes, make([]T, opt.columns-len(sample.Attributes))...)
			}
		}
		samples = append(samples, sample)
	}
	for len(samples) < opt.rows {
		samples = append(samples, stat.Sample[T]{
			Attributes: make(tensor.Vector[T], opt.columns),
		})
	}
	return samples, nil
}
