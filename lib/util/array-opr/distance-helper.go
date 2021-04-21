package arrayOpr

import "math"

func L2DistanceAxis1(array1, array2 [][]float64) ([]float64) {
	res := make([]float64, len(array1))
	for i:=0; i<len(array1); i++ {
		res[i] = L2Distance(array1[i], array2[i])
	}
	return res
}
func L2Distance(array1, array2 []float64) float64 {
	res := math.Sqrt(Sum(BitwisePower(BitwiseMinus(array1, array2), 2)))
	return res
}