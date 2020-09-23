package arrayOpr

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
)

//
// Helper functions to allow easier use of Gonum
//
func Ones(r, c int) mat.Matrix {
	one := make([]float64, r*c)
	for i := 0; i < r*c; i++ {
		one[i] = 1
	}
	oneM := mat.NewDense(r, c, one)
	return oneM
}

func Stack(a, b mat.Matrix) mat.Matrix {
	ra, c := a.Dims()
	rb, _ := b.Dims()
	o := mat.NewDense(ra+rb, c, nil)
	o.Stack(a, b)
	return o
}

func Slice(rs, re, cs, ce int, a mat.Matrix) mat.Matrix {
	var new [][]float64
	for i := rs; i < re; i++ {
		new = append(new, mat.Row(nil, i, a)[cs:ce])
	}
	flatten := Flatten2DFloat(new)
	o := mat.NewDense(re-rs, ce-cs, flatten)
	return o
}

func Dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func Apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

func Scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func Multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func Division(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.DivElem(m, n)
	return o
}

func Pow(m mat.Matrix, x float64) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			o.Set(i, j, math.Pow(m.At(i, j), x))
		}
	}
	return o
}

func Add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func AddScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return Add(m, n)
}

func Subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

// randomly generate a float64 array
func RandomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		// data[i] = rand.NormFloat64() * math.Pow(v, -0.5)
		data[i] = dist.Rand()
	}
	return
}

// func addBiasNodeTo(m mat.Matrix, b float64) mat.Matrix {
// 	r, _ := m.Dims()
// 	a := mat.NewDense(r+1, 1, nil)

// 	a.Set(0, 0, b)
// 	for i := 0; i < r; i++ {
// 		a.Set(i+1, 0, m.At(i, 0))
// 	}
// 	return a
// }

func MatToArray(m mat.Matrix) [][]float64 {
	r, c := m.Dims()
	arr := make([][]float64, r)
	for i := 0; i < r; i++ {
		arr[i] = make([]float64, c)
		for j := 0; j < c; j++ {
			arr[i][j] = m.At(i, j)
		}
	}
	return arr
}

// pretty print a Gonum matrix
func MatrixPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

// func save(net Network) {
// 	h, err := os.Create("data/hweights.model")
// 	defer h.Close()
// 	if err == nil {
// 		for i := 0; i < len(net.HiddenWeights); i++ {
// 			net.HiddenWeights[i].MarshalBinaryTo(h)
// 		}
// 	}
// 	o, err := os.Create("data/oweights.model")
// 	defer o.Close()
// 	if err == nil {
// 		net.OutputWeights.MarshalBinaryTo(o)
// 	}
// }

// // load a neural network from file
// func load(net *Network) {
// 	h, err := os.Open("data/hweights.model")
// 	defer h.Close()
// 	if err == nil {
// 		for i := 0; i < len(net.HiddenWeights); i++ {
// 			net.HiddenWeights[i].Reset()
// 			net.HiddenWeights[i].UnmarshalBinaryFrom(h)
// 		}
// 	}
// 	o, err := os.Open("data/oweights.model")
// 	defer o.Close()
// 	if err == nil {
// 		net.OutputWeights.Reset()
// 		net.OutputWeights.UnmarshalBinaryFrom(o)
// 	}
// 	return
// }

// // predict a number from an image
// // image should be 28 x 28 PNG file
// func predictFromImage(net Network, path string) int {
// 	input := dataFromImage(path)
// 	output := net.Predict(input)
// 	matrixPrint(output)
// 	best := 0
// 	highest := 0.0
// 	for i := 0; i < net.outputs; i++ {
// 		if output.At(i, 0) > highest {
// 			best = i
// 			highest = output.At(i, 0)
// 		}
// 	}
// 	return best
// }

//// get the pixel data from an image
//func dataFromImage(filePath string) (pixels []float64) {
//	// read the file
//	imgFile, err := os.Open(filePath)
//	defer imgFile.Close()
//	if err != nil {
//		fmt.Println("Cannot read file:", err)
//	}
//	img, err := png.Decode(imgFile)
//	if err != nil {
//		fmt.Println("Cannot decode file:", err)
//	}
//
//	// create a grayscale image
//	bounds := img.Bounds()
//	gray := image.NewGray(bounds)
//
//	for x := 0; x < bounds.Max.X; x++ {
//		for y := 0; y < bounds.Max.Y; y++ {
//			var rgba = img.At(x, y)
//			gray.Set(x, y, rgba)
//		}
//	}
//	// make a pixel array
//	pixels = make([]float64, len(gray.Pix))
//	// populate the pixel array subtract Pix from 255 because that's how
//	// the MNIST database was trained (in reverse)
//	for i := 0; i < len(gray.Pix); i++ {
//		pixels[i] = (float64(255-gray.Pix[i]) / 255.0 * 0.999) + 0.001
//	}
//	return
//}
