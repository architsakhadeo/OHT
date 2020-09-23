package optimizer

import (
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"gonum.org/v1/gonum/mat"
	"math"
)

type Optimizer interface {
	Init(alpha float64, momentums []float64, input int, hidden []int, output int)
	Backward(lossMat, OutputWeights mat.Matrix, LayerOut, HiddenWeights []mat.Matrix, tanhLast bool) (mat.Matrix, []mat.Matrix)
}


func reluPrime(r, c int, z float64) float64 {
	if z > 0 {
		return 1
	} else {
		return float64(0)
	}
}

func tanhPrime(r, c int, z float64) float64 {
	return 1 - math.Pow(z, 2)
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return ao.Multiply(m, ao.Subtract(ones, m)) // m * (1 - m)
}
