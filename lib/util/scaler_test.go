package util

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestScalingInfoSimple(t *testing.T) {
	tests := map[string]struct {
		Scaler
		input   []float64
		results []float64
	}{
		"No transformation": {
			NewScaler(0, 10, 10),
			[]float64{0, 3, 10},
			[]float64{0, 3, 10},
		},
		"Scaler to negative": {
			NewScaler(-10, 0, 10),
			[]float64{-10, -7, 0},
			[]float64{0, 3, 10},
		},
		"Double range": {
			NewScaler(0, 10, 20),
			[]float64{0, 3, 10},
			[]float64{0, 6, 20},
		},
		"Half range": {
			NewScaler(0, 10, 5),
			[]float64{0, 3, 10},
			[]float64{0, 1.5, 5},
		},
		"Fractional": {
			NewScaler(0, 8, 1),
			[]float64{0, 2, 4},
			[]float64{0, 0.25, 0.5},
		},
	}

	// Since the Scaler only guarantees the range of the resulting data (not the offset), we need to scale all values so
	// we can test behavior regardless of that offset. So test.results assumes an offset of 0, and the test subtracts
	// the first returned value from all others to match that expected offset of 0.
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			offset := test.Scaler.Scale(test.input[0])
			for i, d := range test.input {
				val := test.Scaler.Scale(d) - offset
				assert.Equal(t, test.results[i], val)
			}
		})
	}
}
