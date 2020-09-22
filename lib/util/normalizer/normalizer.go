package normalizer

type Normalizer struct {
	ArrLen   int
	ArrRange []float64
}

func (nml *Normalizer) MeanZeroNormalization(ori []float64) []float64 {
	new := make([]float64, nml.ArrLen)
	for i := 0; i < nml.ArrLen; i++ {
		new[i] = ori[i] / nml.ArrRange[i] * 2 // if state range is -1 to 1, StateRange returns 2. Thus use *2 to normalize state to correct range
	}
	return new
}
