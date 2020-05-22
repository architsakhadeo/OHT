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
		max, _ := ArrayMax(array[i])
		new[i] = max
	}
	return new
}

func ArrayMax(array []float32) (float32, int){
	var max float32
	var idx int
	for j:=1; j<len(array); j++ {
		if array[j] > max {
			max = array[j]
			idx = j
		}
	}
	return max, idx
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

func A64ToInt(array []float64) []int {
	var aInt []int
    for i, f64 := range array {
        aInt[i] = int(f64)
	}
	return aInt
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
