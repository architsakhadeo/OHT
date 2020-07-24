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
	inputs  int
	hiddens []int
	outputs int
	// HiddenWeights []*mat.Dense
	// OutputWeights *mat.Dense
	HiddenWeights  []mat.Matrix
	OutputWeights  mat.Matrix
	HiddenUpdate   []mat.Matrix
	OutputUpdate   mat.Matrix
	HiddenV        []mat.Matrix
	OutputV        mat.Matrix
	HiddenMomentum []mat.Matrix
	OutputMomentum mat.Matrix
	learningRate   float64
	decay          float64
	SgdMomentum    float64
	beta1          float64
	beta2          float64
	eps            float64

	LayerOut  []mat.Matrix
	iteration int
}

// CreateNetwork creates a neural network with random weights
func CreateNetwork(input int, hidden []int, output int, rate float64, decay float64, SgdMomentum float64, beta1 float64, beta2 float64, eps float64) (net Network) {
	net = Network{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
		learningRate: rate,
		decay:        decay,
		SgdMomentum:  SgdMomentum,
		beta1:        beta1,
		beta2:        beta2,
		eps:          eps,
	}
	ipt := net.inputs
	for i := 0; i < len(net.hiddens); i++ {
		net.HiddenWeights = append(net.HiddenWeights,
			mat.NewDense(net.hiddens[i], ipt,
				randomArray(ipt*net.hiddens[i],
					float64(ipt))))
		// net.HiddenWeights = append(net.HiddenWeights,
		// 	mat.NewDense(net.hiddens[i], ipt, nil))
		ipt = net.hiddens[i] + 1
	}
	net.OutputWeights = mat.NewDense(net.outputs, ipt,
		randomArray(ipt*net.outputs,
			float64(ipt)))
	// net.OutputWeights = mat.NewDense(net.outputs, ipt, nil)

	net.iteration = 0

	lastOut := input
	for i := 0; i < len(hidden); i++ {
		net.HiddenV = append(net.HiddenV, mat.NewDense(hidden[i], lastOut, nil))
		lastOut = hidden[i] + 1
	}
	net.OutputV = mat.NewDense(output, lastOut, nil)

	lastOut = input
	for i := 0; i < len(hidden); i++ {
		net.HiddenMomentum = append(net.HiddenMomentum, mat.NewDense(hidden[i], lastOut, nil))
		lastOut = hidden[i] + 1
	}
	net.OutputMomentum = mat.NewDense(output, lastOut, nil)

	lastOut = input
	for i := 0; i < len(hidden); i++ {
		net.HiddenUpdate = append(net.HiddenUpdate, mat.NewDense(hidden[i], lastOut, nil))
		lastOut = hidden[i] + 1
	}
	net.OutputUpdate = mat.NewDense(output, lastOut, nil)
	return
}

// Forward without gradient
func (net *Network) Predict(inputData [][]float64) [][]float64 {
	flatten := ao.Flatten2DFloat(inputData)
	ipt := mat.NewDense(len(inputData), len(inputData[0]), flatten).T()
	_, c := ipt.Dims()
	for i := 0; i < len(net.HiddenWeights); i++ {
		ipt = dot(net.HiddenWeights[i], ipt)
		ipt = apply(relu, ipt)
		_, c = ipt.Dims()
		ipt = stack(ipt, ones(1, c))
	}
	finalOutputs := dot(net.OutputWeights, ipt)
	finalOutputs = finalOutputs.T()

	arrayOut := matToArray(finalOutputs)
	return arrayOut
}

// Forward with gradient
func (net *Network) Forward(inputData [][]float64) [][]float64 {
	flatten := ao.Flatten2DFloat(inputData)
	ipt := mat.NewDense(len(inputData), len(inputData[0]), flatten).T()
	_, c := ipt.Dims()

	var layerOutputs []mat.Matrix
	layerOutputs = append(layerOutputs, ipt)

	for i := 0; i < len(net.HiddenWeights); i++ {
		ipt = dot(net.HiddenWeights[i], ipt)
		ipt = apply(relu, ipt)
		_, c = ipt.Dims()
		ipt = stack(ipt, ones(1, c))
		layerOutputs = append(layerOutputs, ipt)
	}
	finalOutputs := dot(net.OutputWeights, ipt)
	// layerOutputs = append(layerOutputs, finalOutputs)
	net.LayerOut = layerOutputs
	finalOutputs = finalOutputs.T()

	arrayOut := matToArray(finalOutputs)
	return arrayOut
}

func (net *Network) AdamUpdate(lossMat, lastOut, weight, oldM, oldV mat.Matrix) mat.Matrix {
	var grad, m, v, mHat, vHat mat.Matrix
	grad = scale(-1, dot(lossMat, lastOut.T()))
	m = add(scale(net.beta1, oldM), scale(1-net.beta1, grad))
	v = add(scale(net.beta2, oldV), scale(1-net.beta2, multiply(grad, grad)))
	ro, co := m.Dims()
	mHat = scale(1.0/(1-math.Pow(net.beta1, float64(net.iteration+1))), m)
	vHat = scale(1.0/(1-math.Pow(net.beta2, float64(net.iteration+1))), v)
	// fmt.Println("div ones", ones(ro, co), net.eps)
	// fmt.Println("div scale", scale(net.eps, ones(ro, co)))
	weight = subtract(weight, scale(net.learningRate, division(mHat, add(sqrtEle(vHat), scale(net.eps, ones(ro, co)))))) //.(*mat.Dense)
	// fmt.Println("--", net.iteration+1, net.beta1, net.beta2, oldM, oldV, grad, lossMat, lastOut)
	return weight
}

// gradient and backward
// func (net *Network) Backward(loss float64) {
func (net *Network) Backward(loss [][]float64) {

	flatten := ao.Flatten2DFloat(loss)
	lossMat := mat.NewDense(len(loss), len(loss[0]), flatten).T()

	// r, c := net.LayerOut[len(net.LayerOut)-1].Dims()
	// lossMat := scale(loss, ones(r, c))

	// lr := net.learningRate

	var noBias, mulM mat.Matrix
	var hr, hc, mr, mc int

	hiddenE := make([]mat.Matrix, len(net.HiddenWeights))

	if len(net.HiddenWeights) > 0 {
		hiddenE[len(net.HiddenWeights)-1] = dot(net.OutputWeights.T(), lossMat)
		hr, hc = hiddenE[len(net.HiddenWeights)-1].Dims()
		noBias = slice(0, hr-1, 0, hc, hiddenE[len(net.HiddenWeights)-1])
	}

	for i := len(net.HiddenWeights) - 2; i >= 0; i-- {
		hiddenE[i] = dot(net.HiddenWeights[i+1].T(), noBias)
		hr, hc = hiddenE[i].Dims()
		noBias = slice(0, hr-1, 0, hc, hiddenE[i])
	}

	// var hiddenUpdate mat.Matrix
	// var outputUpdate mat.Matrix

	// fmt.Println("Loss")
	// matrixPrint(lossMat)
	// fmt.Println("Weight")
	// matrixPrint(net.OutputWeights)
	// fmt.Println("LR:", net.learningRate)

	// // ADAM
	// net.OutputWeights = net.AdamUpdate(lossMat, net.LayerOut[len(net.HiddenWeights)], net.OutputWeights,
	// 	net.OutputMomentum, net.OutputV)

	// SGD
	net.OutputUpdate = add(scale(net.SgdMomentum, net.OutputUpdate), scale(net.learningRate, dot(lossMat, net.LayerOut[len(net.HiddenWeights)].T())))
	net.OutputWeights = add(net.OutputWeights, net.OutputUpdate) //.(*mat.Dense)

	// fmt.Println("hiddenOut")
	// matrixPrint(net.LayerOut[len(net.HiddenWeights)].T())
	// // fmt.Println("Update")
	// // matrixPrint(outputUpdate)
	// fmt.Println("Weight")
	// matrixPrint(net.OutputWeights)

	for i := len(net.HiddenWeights) - 1; i >= 0; i-- {
		mulM = multiply(hiddenE[i], apply(reluPrime, net.LayerOut[i+1]))
		mr, mc = mulM.Dims()

		// // ADAM
		// net.HiddenWeights[i] = net.AdamUpdate(slice(0, mr-1, 0, mc, mulM), net.LayerOut[i], net.HiddenWeights[i],
		// 	net.HiddenMomentum[i], net.HiddenV[i])

		// SGD
		net.HiddenUpdate[i] = add(scale(net.SgdMomentum, net.HiddenUpdate[i]), scale(net.learningRate, dot(slice(0, mr-1, 0, mc, mulM), net.LayerOut[i].T()))) //.(*mat.Dense)
		net.HiddenWeights[i] = add(net.HiddenWeights[i], net.HiddenUpdate[i])                                                                                  //.(*mat.Dense)
	}
	net.LayerOut = nil
	net.iteration = net.iteration + 1
	// fmt.Println()

}

func relu(r, c int, z float64) float64 {
	if z >= 0 {
		return z
	} else {
		return float64(0)
	}
}

func reluPrime(r, c int, z float64) float64 {
	if z > 0 {
		return 1
	} else {
		return float64(0)
	}
}

// func sigmoid(r, c int, z float64) float64 {
// 	return 1.0 / (1 + math.Exp(-1*z))
// }

// func sigmoidPrime(m mat.Matrix) mat.Matrix {
// 	rows, _ := m.Dims()
// 	o := make([]float64, rows)
// 	for i := range o {
// 		o[i] = 1
// 	}
// 	ones := mat.NewDense(rows, 1, o)
// 	return multiply(m, subtract(ones, m)) // m * (1 - m)
// }

//
// Helper functions to allow easier use of Gonum
//
func ones(r, c int) mat.Matrix {
	one := make([]float64, r*c)
	for i := 0; i < r*c; i++ {
		one[i] = 1
	}
	oneM := mat.NewDense(r, c, one)
	return oneM
}

func stack(a, b mat.Matrix) mat.Matrix {
	ra, c := a.Dims()
	rb, _ := b.Dims()
	o := mat.NewDense(ra+rb, c, nil)
	o.Stack(a, b)
	return o
}

func slice(rs, re, cs, ce int, a mat.Matrix) mat.Matrix {
	var new [][]float64
	for i := rs; i < re; i++ {
		new = append(new, mat.Row(nil, i, a)[cs:ce])
	}
	flatten := ao.Flatten2DFloat(new)
	o := mat.NewDense(re-rs, ce-cs, flatten)
	return o
}

func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
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

func division(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.DivElem(m, n)
	return o
}

func sqrtEle(m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			o.Set(i, j, math.Sqrt(m.At(i, j)))
		}
	}
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

// func addBiasNodeTo(m mat.Matrix, b float64) mat.Matrix {
// 	r, _ := m.Dims()
// 	a := mat.NewDense(r+1, 1, nil)

// 	a.Set(0, 0, b)
// 	for i := 0; i < r; i++ {
// 		a.Set(i+1, 0, m.At(i, 0))
// 	}
// 	return a
// }

func matToArray(m mat.Matrix) [][]float64 {
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
func matrixPrint(X mat.Matrix) {
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
