package network

// This file is based on https://github.com/sausheong/gonn

import (
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"github.com/stellentus/cartpoles/lib/util/optimizer"
	"gonum.org/v1/gonum/mat"
	"math"
)

// Network is a neural network with 3 layers
type Network struct {
	inputs  int
	hiddens []int
	outputs int

	HiddenWeights  []mat.Matrix
	OutputWeights  mat.Matrix
	HiddenUpdate   []mat.Matrix
	OutputUpdate   mat.Matrix

	//HiddenV        []mat.Matrix
	//OutputV        mat.Matrix
	//HiddenMomentum []mat.Matrix
	//OutputMomentum mat.Matrix

	tanhLast 		bool
	//learningRate   float64
	//decay          float64
	//SgdMomentum    float64
	//beta1          float64
	//beta2          float64
	//eps            float64

	LayerOut  []mat.Matrix
	//iteration int
}

// CreateNetwork creates a neural network with random weights
func CreateNetwork(input int, hidden []int, output int, rate float64, decay float64, SgdMomentum float64, beta1 float64, beta2 float64, eps float64) (net Network) {
	net = Network{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
		//learningRate: rate,
		//decay:        decay,
		//SgdMomentum:  SgdMomentum,
		//beta1:        beta1,
		//beta2:        beta2,
		//eps:          eps,
	}
	ipt := net.inputs
	for i := 0; i < len(net.hiddens); i++ {
		net.HiddenWeights = append(net.HiddenWeights,
			mat.NewDense(net.hiddens[i], ipt,
				ao.RandomArray(ipt*net.hiddens[i],
					float64(ipt))))
		ipt = net.hiddens[i] + 1
	}
	net.OutputWeights = mat.NewDense(net.outputs, ipt,
		ao.RandomArray(ipt*net.outputs,
			float64(ipt)))

	//net.iteration = 0

	//lastOut := input
	//for i := 0; i < len(hidden); i++ {
	//	net.HiddenV = append(net.HiddenV, mat.NewDense(hidden[i], lastOut, nil))
	//	lastOut = hidden[i] + 1
	//}
	//net.OutputV = mat.NewDense(output, lastOut, nil)
	//
	//lastOut = input
	//for i := 0; i < len(hidden); i++ {
	//	net.HiddenMomentum = append(net.HiddenMomentum, mat.NewDense(hidden[i], lastOut, nil))
	//	lastOut = hidden[i] + 1
	//}
	//net.OutputMomentum = mat.NewDense(output, lastOut, nil)

	lastOut := input
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
		ipt = ao.Dot(net.HiddenWeights[i], ipt)
		ipt = ao.Apply(relu, ipt)
		_, c = ipt.Dims()
		ipt = ao.Stack(ipt, ao.Ones(1, c))
	}
	finalOutputs := ao.Dot(net.OutputWeights, ipt)
	if net.tanhLast {
		finalOutputs = ao.Apply(tanh, ipt)
	}
	finalOutputs = finalOutputs.T()

	arrayOut := ao.MatToArray(finalOutputs)
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
		ipt = ao.Dot(net.HiddenWeights[i], ipt)
		ipt = ao.Apply(relu, ipt)
		_, c = ipt.Dims()
		ipt = ao.Stack(ipt, ao.Ones(1, c))
		layerOutputs = append(layerOutputs, ipt)
	}
	finalOutputs := ao.Dot(net.OutputWeights, ipt)
	if net.tanhLast {
		finalOutputs = ao.Apply(tanh, ipt)
	}
	net.LayerOut = layerOutputs
	finalOutputs = finalOutputs.T()

	arrayOut := ao.MatToArray(finalOutputs)
	return arrayOut
}

//func (net *Network) AdamUpdate(lossMat, lastOut, weight, oldM, oldV mat.Matrix) (mat.Matrix, mat.Matrix, mat.Matrix) {
//	var grad, m, v mat.Matrix
//	var mHat, vHat mat.Matrix
//	grad = scale(-1, dot(lossMat, lastOut.T()))
//	m = add(scale(net.beta1, oldM), scale(1-net.beta1, grad))
//	v = add(scale(net.beta2, oldV), scale(1-net.beta2, pow(grad, 2.0)))
//	ro, co := m.Dims()
//	mHat = scale(1.0/(1-math.Pow(net.beta1, float64(net.iteration+1))), m)
//	vHat = scale(1.0/(1-math.Pow(net.beta2, float64(net.iteration+1))), v)
//	weight = subtract(weight,
//		scale(net.learningRate, division(mHat, add(pow(vHat, 0.5), scale(net.eps, ones(ro, co))))))
//	return weight, m, v
//}

// gradient and backward
func (net *Network) Backward(loss [][]float64, opt optimizer.Optimizer) {
	flatten := ao.Flatten2DFloat(loss)
	lossMat := mat.NewDense(len(loss), len(loss[0]), flatten).T()
	net.OutputWeights, net.HiddenWeights = opt.Backward(lossMat, net.OutputWeights, net.LayerOut, net.HiddenWeights, net.tanhLast)
	net.LayerOut = nil
}

func relu(r, c int, z float64) float64 {
	if z >= 0 {
		return z
	} else {
		return float64(0)
	}
}

func tanh(r, c int, z float64) float64 {
	exp := math.Exp(z)
	expNeg := math.Exp(-1 * z)
	return (exp - expNeg) / (exp + expNeg)
}

func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func Synchronization(from, to Network) Network {
	for i := 0; i < len(to.HiddenWeights); i++ {
		to.HiddenWeights[i] = from.HiddenWeights[i]
	}
	to.OutputWeights = from.OutputWeights
	return to
}

