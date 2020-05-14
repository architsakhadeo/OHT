package util

import (
	"math"
)

// Scaler is used to provide a scaling range for tiles.
type Scaler struct {
	sub, mult float64
	MaxRange  int
}

// NewScaler creates a new Scaler struct.
// It's used to indicate that data should be scaled from [min, max] to the range [0+x, maxRange+x], where x is an unspecified integer.
func NewScaler(min, max float64, maxRange int) Scaler {
	mult := float64(maxRange) / (max - min)
	sub := min * mult
	_, fracMin := math.Modf(min)
	_, fracMult := math.Modf(mult)
	_, fracSub := math.Modf(sub)
	if (fracMin == 0 && fracMult == 0) || fracSub == 0 {
		// If 'sub' is an integer, then we don't actually need to include it in the scaling, since we don't care about the offset of the values.
		sub = 0
	}

	return Scaler{sub: sub, mult: mult, MaxRange: maxRange}
}

// Scaler scales a value based on the Scaler settings.
// If the input data is below min or above max, the behavior is not defined (but it will probably mess up your indices).
func (scale Scaler) Scale(val float64) float64 {
	return val*scale.mult - scale.sub
}
