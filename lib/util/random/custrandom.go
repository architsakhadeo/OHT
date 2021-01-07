package random

import (
	"math/rand"
)

func FreqSample(cdf []float64) int {
	r := rand.Float64()
	bucket := 0
	for r > cdf[bucket] {
		bucket++
	}
	return bucket
}
