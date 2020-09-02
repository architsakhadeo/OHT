package arrayOpr

import (
	"math"

	"github.com/stellentus/cartpoles/lib/rlglue"
)

func A32Col(array [][]float32, col int) []float32 {
	var new []float32
	for i := 0; i < len(array); i++ {
		// new[i] = array[i][col]
		new = append(new, array[i][col])
	}
	return new
}

/*
Take block from 2-d array
*/
func Index2d(array [][]float64, rowStart int, rowEnd int, colStart int, colEnd int) [][]float64 {
	var new [][]float64
	for i := rowStart; i < rowEnd; i++ {
		var temp []float64
		new = append(new, temp)
		for j := colStart; j < colEnd; j++ {
			// new[i-rowStart][j-colStart] = array[i][j]
			new[i] = append(new[i], array[i][j])
		}
	}
	return new
}

/*
Index in each row of 2-d array
*/
func RowIndexFloat(array [][]float64, idx []int) []float64 {
	// var new []float64
	// for i := 0; i < len(array); i++ {
	// 	new = append(new, array[i][idx[i]])
	// }
	new := make([]float64, len(array))
	for i := 0; i < len(array); i++ {
		new[i] = array[i][idx[i]]
	}
	return new
}

/*
Max in each row of 2-d array
*/
func RowIndexMax(array [][]float64) ([]float64, []int) {
	new := make([]float64, len(array))
	arg := make([]int, len(array))
	for i := 0; i < len(array); i++ {
		max, idx := ArrayMax(array[i])
		new[i] = max
		arg[i] = idx
	}
	return new, arg
}

/*
Max in each row of 1-d array
*/
func ArrayMax(array []float64) (float64, int) {
	max := math.Inf(-1)
	var idx int
	for j := 0; j < len(array); j++ {
		if array[j] > max {
			max = array[j]
			idx = j
		}
	}
	return max, idx
}

func StateTo32(state rlglue.State) []float32 {
	var a32 []float32
	for i := 0; i < len(state); i++ {
		a32 = append(a32, float32(state[i]))
	}
	return a32
}

func A64To32_2d(array [][]float64) [][]float32 {
	var a32 [][]float32
	for i := 0; i < len(array); i++ {
		var temp []float32
		a32 = append(a32, temp)
		for j := 0; j < len(array[i]); j++ {
			a32[i] = append(a32[i], float32(array[i][j]))
		}
	}
	return a32
}

func A64To32(array []float64) []float32 {
	var a32 []float32
	for i, f64 := range array {
		a32[i] = float32(f64)
	}
	return a32
}

func A32To64(array []float32) []float64 {
	var a64 []float64
	for i, f32 := range array {
		a64[i] = float64(f32)
	}
	return a64
}

func A32ToInt(array []float32) []int {
	var aInt []int
	for i, f32 := range array {
		aInt[i] = int(f32)
	}
	return aInt
}

func A64ToInt(array []float64) []int {
	var aInt []int
	for _, f64 := range array {
		aInt = append(aInt, int(f64))
	}
	return aInt
}
func A64ToInt2D(array [][]float64) [][]int {
	var aInt [][]int
	for i := 0; i < len(array); i++ {
		aInt = append(aInt, A64ToInt(array[i]))
	}
	return aInt
}

func BitwiseAdd(a []float32, b []float32) []float32 {
	var res []float32
	// if len(a) != len(b) {
	// 	errors.New("Arrays should have same length")
	// }
	for i := 0; i < len(a); i++ {
		res[i] = a[i] + b[i]
	}
	return res
}

func BitwiseMulti(a []float32, b []float32) []float32 {
	var res []float32
	// if len(a) != len(b) {
	// 	errors.New("Arrays should have same length")
	// }
	for i := 0; i < len(a); i++ {
		res[i] = a[i] * b[i]
	}
	return res
}

func BitwiseMinus(a []float32, b []float32) []float32 {
	var res []float32
	// if len(a) != len(b) {
	// 	errors.New("Arrays should have same length")
	// }
	for i := 0; i < len(a); i++ {
		res[i] = a[i] - b[i]
	}
	return res
}

func BitwiseDivide(a []float32, b []float32) []float32 {
	var res []float32
	// if len(a) != len(b) {
	// 	errors.New("Arrays should have same length")
	// }
	for i := 0; i < len(a); i++ {
		res[i] = a[i] / b[i]
	}
	return res
}

func Flatten2DFloat(inputData [][]float64) []float64 {
	var flatten []float64
	for i := 0; i < len(inputData); i++ {
		for j := 0; j < len(inputData[0]); j++ {
			flatten = append(flatten, inputData[i][j])
		}
	}
	return flatten
}

func Flatten2DInt(inputData [][]int) []int {
	var flatten []int
	for i := 0; i < len(inputData); i++ {
		for j := 0; j < len(inputData[0]); j++ {
			flatten = append(flatten, inputData[i][j])
		}
	}
	return flatten
}

func Average(inputData []float64) float64{
	sum := 0.0
	for i := 0; i < len(inputData); i++ {
		sum = sum + inputData[i]
	}
	sum = sum / float64(len(inputData))
	return sum
}