package optimizer

import (
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"gonum.org/v1/gonum/mat"
)

type Sgd struct {
	alpha     float64
	momentum  float64
	iteration int

	HiddenUpdate        []mat.Matrix
	OutputUpdate        mat.Matrix
}

func (opt *Sgd) Init(alpha float64, momentum []float64, input int, hidden []int, output int) {
	opt.alpha = alpha
	opt.momentum = momentum[0]
	opt.iteration = 0

	lastOut := input
	for i := 0; i < len(hidden); i++ {
		opt.HiddenUpdate = append(opt.HiddenUpdate, mat.NewDense(hidden[i], lastOut, nil))
		lastOut = hidden[i] + 1
	}
	opt.OutputUpdate = mat.NewDense(output, lastOut, nil)
	return
}

func (opt *Sgd) Backward(lossMat, OutputWeights mat.Matrix, LayerOut, HiddenWeights []mat.Matrix, tanhLast bool) (mat.Matrix, []mat.Matrix){

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

	opt.OutputUpdate = ao.Add(ao.Scale(opt.momentum, opt.OutputUpdate), ao.Scale(opt.alpha, ao.Dot(lossMat, LayerOut[len(HiddenWeights)].T())))
	OutputWeights = ao.Add(OutputWeights, opt.OutputUpdate) //.(*mat.Dense)

	for i := len(HiddenWeights) - 1; i >= 0; i-- {
		mulM = ao.Multiply(hiddenE[i], ao.Apply(reluPrime, LayerOut[i+1]))
		mr, mc = mulM.Dims()

		opt.HiddenUpdate[i] = ao.Add(ao.Scale(opt.momentum, opt.HiddenUpdate[i]), ao.Scale(opt.alpha, ao.Dot(ao.Slice(0, mr-1, 0, mc, mulM), LayerOut[i].T()))) //.(*mat.Dense)
		HiddenWeights[i] = ao.Add(HiddenWeights[i], opt.HiddenUpdate[i])                                                                                  //.(*mat.Dense)
	}
	opt.iteration = opt.iteration + 1

	return OutputWeights, HiddenWeights
}
