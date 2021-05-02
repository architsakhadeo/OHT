package loss

import (
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
)

func MseLoss(target, predict [][]float64) float64 {
	loss := ao.BitwisePower2D(ao.BitwiseMinus2D(target, predict), 2.0)

	avgLoss := make([][]float64, 1)
	avgLoss[0] = make([]float64, len(loss[0]))
	for j := 0; j < len(loss[0]); j++ {
		sum := 0.0
		for i := 0; i < len(loss); i++ {
			sum += loss[i][j]
		}
		avgLoss[0][j] = sum / float64(len(loss))
	}
	return ao.Average(ao.Flatten2DFloat(avgLoss))
}

func MseLossDeriv(target, predict [][]float64) [][]float64 {
	deriv := ao.BitwiseMinus2D(predict, target)
	//return deriv
	avgDeriv := make([][]float64, len(target))
	for i:=0; i<len(target); i++ {
		avgDeriv[i] = make([]float64, len(target[0]))
	}
	for j := 0; j < len(deriv[0]); j++ {
		sum := 0.0
		for i := 0; i < len(deriv); i++ {
			sum += deriv[i][j]
		}
		//avgDeriv[0][j] = sum / float64(len(deriv))
		avg := sum / float64(len(deriv))
		for i:=0; i<len(target); i++ {
			avgDeriv[i][j] = avg
		}
	}
	return avgDeriv
}

