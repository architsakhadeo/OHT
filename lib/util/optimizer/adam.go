package optimizer

import (
	"math"

	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"gonum.org/v1/gonum/mat"
)

type Adam struct {
	alpha     float64
	beta1     float64
	beta2     float64
	eps       float64
	lambda    float64 // weight decay coefficient
	iteration int

	HiddenV        []mat.Matrix
	OutputV        mat.Matrix
	HiddenMomentum []mat.Matrix
	OutputMomentum mat.Matrix
}

func (opt *Adam) Init(alpha float64, momentums []float64, input int, hidden []int, output int) {
	opt.alpha = alpha
	opt.beta1 = momentums[0]
	opt.beta2 = momentums[1]
	opt.eps = momentums[2]
	if len(momentums) == 4 {
		// Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
		opt.lambda = momentums[3]
	}
	opt.iteration = 0

	lastOut := input
	for i := 0; i < len(hidden); i++ {
		opt.HiddenV = append(opt.HiddenV, mat.NewDense(hidden[i], lastOut, nil))
		lastOut = hidden[i] + 1
	}
	opt.OutputV = mat.NewDense(output, lastOut, nil)

	lastOut = input
	for i := 0; i < len(hidden); i++ {
		opt.HiddenMomentum = append(opt.HiddenMomentum, mat.NewDense(hidden[i], lastOut, nil))
		lastOut = hidden[i] + 1
	}
	opt.OutputMomentum = mat.NewDense(output, lastOut, nil)
	return
}

func (opt *Adam) AdamUpdate(lossMat, lastOut, weight, oldM, oldV mat.Matrix) (mat.Matrix, mat.Matrix, mat.Matrix) {
	var grad, m, v mat.Matrix
	var mHat, vHat mat.Matrix
	grad = ao.Scale(-1, ao.Dot(lossMat, lastOut.T()))
	m = ao.Add(ao.Scale(opt.beta1, oldM), ao.Scale(1-opt.beta1, grad))
	v = ao.Add(ao.Scale(opt.beta2, oldV), ao.Scale(1-opt.beta2, ao.Pow(grad, 2.0)))
	ro, co := m.Dims()
	mHat = ao.Scale(1.0/(1-math.Pow(opt.beta1, float64(opt.iteration+1))), m)
	vHat = ao.Scale(1.0/(1-math.Pow(opt.beta2, float64(opt.iteration+1))), v)
	weight = ao.Subtract(weight,
		ao.Add(
			ao.Scale(opt.alpha, ao.Division(mHat, ao.Add(ao.Pow(vHat, 0.5), ao.Scale(opt.eps, ao.Ones(ro, co))))),
			ao.Scale(opt.lambda, weight)))
	return weight, m, v
}

func (opt *Adam) Backward(lossMat, OutputWeights mat.Matrix, LayerOut, HiddenWeights []mat.Matrix, tanhLast bool) (mat.Matrix, []mat.Matrix) {

	var noBias, mulM mat.Matrix
	var hr, hc, mr, mc int

	hiddenE := make([]mat.Matrix, len(HiddenWeights))

	if len(HiddenWeights) > 0 {
		hiddenE[len(HiddenWeights)-1] = ao.Dot(OutputWeights.T(), lossMat)
		hr, hc = hiddenE[len(HiddenWeights)-1].Dims()
		noBias = ao.Slice(0, hr-1, 0, hc, hiddenE[len(HiddenWeights)-1])
	}

	for i := len(HiddenWeights) - 2; i >= 0; i-- {
		hiddenE[i] = ao.Dot(HiddenWeights[i+1].T(), noBias)
		hr, hc = hiddenE[i].Dims()
		noBias = ao.Slice(0, hr-1, 0, hc, hiddenE[i])
	}

	if tanhLast {
		lossMat = ao.Apply(tanhPrime, lossMat)
	}

	//ADAM
	OutputWeights, opt.OutputMomentum, opt.OutputV = opt.AdamUpdate(lossMat, LayerOut[len(HiddenWeights)],
		OutputWeights, opt.OutputMomentum, opt.OutputV)

	//// SGD
	//net.OutputUpdate = add(scale(net.SgdMomentum, net.OutputUpdate), scale(net.learningRate, dot(lossMat, net.LayerOut[len(net.HiddenWeights)].T())))
	//net.OutputWeights = add(net.OutputWeights, net.OutputUpdate) //.(*mat.Dense)

	for i := len(HiddenWeights) - 1; i >= 0; i-- {
		mulM = ao.Multiply(hiddenE[i], ao.Apply(reluPrime, LayerOut[i+1]))
		mr, mc = mulM.Dims()

		// ADAM
		HiddenWeights[i], opt.HiddenMomentum[i], opt.HiddenV[i] = opt.AdamUpdate(ao.Slice(0, mr-1, 0, mc, mulM),
			LayerOut[i], HiddenWeights[i], opt.HiddenMomentum[i], opt.HiddenV[i])

		//// SGD
		//net.HiddenUpdate[i] = add(scale(net.SgdMomentum, net.HiddenUpdate[i]), scale(net.learningRate, dot(slice(0, mr-1, 0, mc, mulM), net.LayerOut[i].T()))) //.(*mat.Dense)
		//net.HiddenWeights[i] = add(net.HiddenWeights[i], net.HiddenUpdate[i])                                                                                  //.(*mat.Dense)
	}
	opt.iteration = opt.iteration + 1

	return OutputWeights, HiddenWeights
}
