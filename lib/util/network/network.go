package network

// This file is based on https://github.com/sausheong/gonn

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"

	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
)

// Network is a neural network with 3 layers
type Network struct {
	inputs        int
	hiddens       []int
	outputs       int
	HiddenWeights []*mat.Dense
	OutputWeights *mat.Dense
	learningRate  float64

	layerOut []mat.Matrix
}

// CreateNetwork creates a neural network with random weights
func CreateNetwork(input int, hidden []int, output int, rate float64) (net Network) {
	net = Network{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
		learningRate: rate,
	}
	ipt := net.inputs
	for i := 0; i < len(net.hiddens); i++ {
		net.HiddenWeights = append(net.HiddenWeights, mat.NewDense(net.hiddens[i], ipt, randomArray(ipt*net.hiddens[i], float64(ipt))))
		ipt = net.hiddens[i]
	}
	net.OutputWeights = mat.NewDense(net.outputs, net.hiddens[len(net.hiddens)-1],
		randomArray(net.hiddens[len(net.hiddens)-1]*net.outputs,
			float64(net.hiddens[len(net.hiddens)-1])))
	return
}

// Forward without gradient
func (net *Network) Predict(inputData [][]float64) mat.Matrix {
	// feedforward
	flatten := ao.Flatten2DFloat(inputData)
	ipt := mat.NewDense(len(inputData), len(inputData[0]), flatten).T()
	for i := 0; i < len(net.HiddenWeights); i++ {
		ipt = dot(net.HiddenWeights[i], ipt)
		ipt = apply(relu, ipt)
	}
	finalOutputs := dot(net.OutputWeights, ipt)
	finalOutputs = finalOutputs.T()
	return finalOutputs
}

// Forward with gradient
func (net *Network) Forward(inputData [][]float64) mat.Matrix {
	flatten := ao.Flatten2DFloat(inputData)
	ipt := mat.NewDense(len(inputData), len(inputData[0]), flatten).T()

	var layerOutputs []mat.Matrix
	layerOutputs = append(layerOutputs, ipt)

	for i := 0; i < len(net.HiddenWeights); i++ {
		ipt = dot(net.HiddenWeights[i], ipt)
		ipt = apply(relu, ipt)
		layerOutputs = append(layerOutputs, ipt)
	}
	finalOutputs := dot(net.OutputWeights, ipt)
	layerOutputs = append(layerOutputs, finalOutputs)
	net.layerOut = layerOutputs

	finalOutputs = finalOutputs.T()
	return finalOutputs
}

// gradient and backward
func (net *Network) Backward(loss float64) {
	// error
	r, c := net.layerOut[len(net.layerOut)-1].Dims()
	lossMat := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			lossMat.Set(i, j, loss)
		}
	}
	// flatten := ao.Flatten2DFloat(loss)
	// lossMat := mat.NewDense(len(loss), len(loss[0]), flatten).T()

	var hiddenErrors []mat.Matrix
	hiddenErrors = append(hiddenErrors, dot(net.OutputWeights.T(), lossMat))

	for i := len(net.HiddenWeights) - 1; i >= 0; i-- {
		hiddenErrors = append(hiddenErrors, dot(net.HiddenWeights[i].T(), hiddenErrors[len(hiddenErrors)-1]))
	}

	net.OutputWeights = add(net.OutputWeights,
		scale(net.learningRate,
			dot(multiply(lossMat, net.layerOut[len(net.layerOut)-1]),
				net.layerOut[len(net.layerOut)-2].T()))).(*mat.Dense)

	for i := len(net.HiddenWeights) - 1; i > 1; i-- {
		net.HiddenWeights[i] = add(net.HiddenWeights[i],
			scale(net.learningRate,
				dot(multiply(hiddenErrors[i-len(net.HiddenWeights)+1], apply(reluPrime, net.layerOut[i-1])),
					net.layerOut[i-2].T()))).(*mat.Dense)
	}
	net.layerOut = nil
}

func relu(r, c int, z float64) float64 {
	if z >= 0 {
		return z
	} else {
		return float64(0)
	}
}

func reluPrime(r, c int, z float64) float64 {
	if z >= 0 {
		return 1
	} else {
		return float64(0)
	}
}

// func sigmoid(r, c int, z float64) float64 {
// 	return 1.0 / (1 + math.Exp(-1*z))
// }

func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(m, subtract(ones, m)) // m * (1 - m)
}

//
// Helper functions to allow easier use of Gonum
//

func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	// r, ct := m.Dims()
	// rt, c := n.Dims()
	// fmt.Println(r, ct, rt, c)
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

func scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func addScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return add(m, n)
}

func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

// randomly generate a float64 array
func randomArray(size int, v float64) (data []float64) {
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

func addBiasNodeTo(m mat.Matrix, b float64) mat.Matrix {
	r, _ := m.Dims()
	a := mat.NewDense(r+1, 1, nil)

	a.Set(0, 0, b)
	for i := 0; i < r; i++ {
		a.Set(i+1, 0, m.At(i, 0))
	}
	return a
}

// pretty print a Gonum matrix
func matrixPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func save(net Network) {
	h, err := os.Create("data/hweights.model")
	defer h.Close()
	if err == nil {
		for i := 0; i < len(net.HiddenWeights); i++ {
			net.HiddenWeights[i].MarshalBinaryTo(h)
		}
	}
	o, err := os.Create("data/oweights.model")
	defer o.Close()
	if err == nil {
		net.OutputWeights.MarshalBinaryTo(o)
	}
}

// load a neural network from file
func load(net *Network) {
	h, err := os.Open("data/hweights.model")
	defer h.Close()
	if err == nil {
		for i := 0; i < len(net.HiddenWeights); i++ {
			net.HiddenWeights[i].Reset()
			net.HiddenWeights[i].UnmarshalBinaryFrom(h)
		}
	}
	o, err := os.Open("data/oweights.model")
	defer o.Close()
	if err == nil {
		net.OutputWeights.Reset()
		net.OutputWeights.UnmarshalBinaryFrom(o)
	}
	return
}

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

// get the pixel data from an image
func dataFromImage(filePath string) (pixels []float64) {
	// read the file
	imgFile, err := os.Open(filePath)
	defer imgFile.Close()
	if err != nil {
		fmt.Println("Cannot read file:", err)
	}
	img, err := png.Decode(imgFile)
	if err != nil {
		fmt.Println("Cannot decode file:", err)
	}

	// create a grayscale image
	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			var rgba = img.At(x, y)
			gray.Set(x, y, rgba)
		}
	}
	// make a pixel array
	pixels = make([]float64, len(gray.Pix))
	// populate the pixel array subtract Pix from 255 because that's how
	// the MNIST database was trained (in reverse)
	for i := 0; i < len(gray.Pix); i++ {
		pixels[i] = (float64(255-gray.Pix[i]) / 255.0 * 0.999) + 0.001
	}
	return
}
