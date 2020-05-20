package arrayOpr

func RowIndexFloat(array [][]float32, idx []int) []float32{
	var new []float32
	for i:=0; i<len(array); i++ {
		new[i] = array[i][idx[i]]
	}
	return new
}

func RowIndexMax(array [][]float32) []float32{
	var new []float32
	for i:=0; i<len(array); i++ {
		max := ArrayMax(array[i])
		new[i] = max
	}
	return new
}

func ArrayMax(array []float32) float32{
	var max float32
	for j:=1; j<len(array); j++ {
		if array[j] > max {
			max = array[j]
		}
	}
	return max
}

func BitwiseAdd(a []float32, b []float32) []float32 {
	var res []float32
	// if len(a) != len(b) {
	// 	errors.New("Arrays should have same length")
	// }
	for i:=0; i<len(a); i++ {
		res[i] = a[i] + b[i]
	}
	return res
}

func BitwiseMulti(a []float32, b []float32) []float32 {
	var res []float32
	// if len(a) != len(b) {
	// 	errors.New("Arrays should have same length")
	// }
	for i:=0; i<len(a); i++ {
		res[i] = a[i] * b[i]
	}
	return res
}

func BitwiseMinus(a []float32, b []float32) []float32 {
	var res []float32
	// if len(a) != len(b) {
	// 	errors.New("Arrays should have same length")
	// }
	for i:=0; i<len(a); i++ {
		res[i] = a[i] - b[i]
	}
	return res
}

func BitwiseDivide(a []float32, b []float32) []float32 {
	var res []float32
	// if len(a) != len(b) {
	// 	errors.New("Arrays should have same length")
	// }
	for i:=0; i<len(a); i++ {
		res[i] = a[i] / b[i]
	}
	return res
}
