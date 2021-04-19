package random

import (
	"math/rand"
)

func FreqSample(normProb []float64) int {
	cdf := make([]float64, len(normProb))
	temp1 := 0.0
	for i := 0; i < len(normProb); i++ {
		cdf[i] = temp1 + normProb[i]
		temp1 = cdf[i]
	}

	r := rand.Float64()
	bucket := 0
	for r > cdf[bucket] {
		bucket++
	}
	return bucket
}
