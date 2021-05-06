package transnetwork

import (
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"github.com/stellentus/cartpoles/lib/util/loss"
	"github.com/stellentus/cartpoles/lib/util/network"
	"github.com/stellentus/cartpoles/lib/util/optimizer"
	"log"
	"math"
	"math/rand"
)

type TransNetwork struct {
	seed	 		int64
	rng 			*rand.Rand
	numAction 		int
	inputStateDim  int
	outputStateDim  int
	hiddenLayer   	[]int
	nnFunc			[]network.Network
	optimizer		[]optimizer.Optimizer
	learningRate   	float64
	batchSize   	int
	numEpoch		int

	inputData		[][]float64
	groundTruth		[][]float64

	dataSet 		[][]float64
	testInputs 		[][]float64
	testTruths 		[][]float64

	convergeCheck	int
	separatedNetwork bool
}

func New() *TransNetwork {
	return &TransNetwork{}
}

func (tn *TransNetwork) Initialize(seed int64, dataSet [][]float64, numEpoch int, batchSize int, learningRate float64, hiddenLayer []int,
	inputStateDim int, outputStateDim int, numAction int, separatedNN bool) {

	tn.seed = seed
	tn.rng = rand.New(rand.NewSource(int64(seed)))
	tn.dataSet = dataSet
	tn.numEpoch = numEpoch
	tn.inputStateDim = inputStateDim
	tn.outputStateDim = outputStateDim
	tn.numAction = numAction
	tn.hiddenLayer = hiddenLayer
	tn.learningRate = learningRate
	tn.batchSize = batchSize
	tn.convergeCheck = 3
	tn.separatedNetwork = separatedNN
	if tn.separatedNetwork {
		tn.nnFunc = make([]network.Network, 3)
		tn.optimizer = make([]optimizer.Optimizer, 3)
	} else {
		tn.nnFunc = make([]network.Network, 1)
		tn.optimizer = make([]optimizer.Optimizer, 1)
	}
	tn.NetworkInit()
}

func (tn *TransNetwork) NetworkInit() {
	inputLen := tn.inputStateDim+tn.numAction
	if tn.separatedNetwork {
		tn.nnFunc[0] = network.CreateNetwork(inputLen, tn.hiddenLayer, tn.outputStateDim, tn.learningRate,
			0, 0, 0.9, 0.999, 1e-08)
		tn.nnFunc[1] = network.CreateNetwork(inputLen, tn.hiddenLayer, 1, tn.learningRate,
			0, 0, 0.9, 0.999, 1e-08)
		tn.nnFunc[2] = network.CreateNetwork(inputLen, tn.hiddenLayer, 1, tn.learningRate,
			0, 0, 0.9, 0.999, 1e-08)
		tn.optimizer[0] = new(optimizer.Adam)
		tn.optimizer[0].Init(tn.learningRate, []float64{0.9, 0.999, 1e-08}, inputLen, tn.hiddenLayer, tn.outputStateDim)
		tn.optimizer[1] = new(optimizer.Adam)
		tn.optimizer[1].Init(tn.learningRate, []float64{0.9, 0.999, 1e-08}, inputLen, tn.hiddenLayer, 1)
		tn.optimizer[2] = new(optimizer.Adam)
		tn.optimizer[2].Init(tn.learningRate, []float64{0.9, 0.999, 1e-08}, inputLen, tn.hiddenLayer, 1)
	} else {
		outputLen := tn.outputStateDim+2
		tn.nnFunc[0] = network.CreateNetwork(inputLen, tn.hiddenLayer, outputLen, tn.learningRate,
			0, 0, 0.9, 0.999, 1e-08)
		tn.optimizer[0] = new(optimizer.Adam)
		tn.optimizer[0].Init(tn.learningRate, []float64{0.9, 0.999, 1e-08}, inputLen, tn.hiddenLayer, outputLen)
	}
}

func (tn *TransNetwork) OrganizeData(dataSet [][]float64) ([][]float64, [][]float64) {
	input := make([][]float64, len(dataSet))
	output := make([][]float64, len(dataSet))

	for i:=0; i<len(dataSet); i++ {
		input[i] = make([]float64, tn.inputStateDim+tn.numAction)
		output[i] = make([]float64, tn.outputStateDim+2)
		action := dataSet[i][tn.inputStateDim]
		oneHotA := ao.OneHotSet(action, tn.numAction)
		copy(input[i][: tn.inputStateDim], dataSet[i][: tn.inputStateDim])
		copy(input[i][tn.inputStateDim: ], oneHotA)
		copy(output[i], dataSet[i][tn.inputStateDim+1: ])
	}

	return input, output
}

func (tn *TransNetwork) Train() []network.Network {
	tn.inputData, tn.groundTruth = tn.OrganizeData(tn.dataSet)
	deriv := make([]float64, len(tn.inputData)/tn.batchSize+1)
	losses := make([]float64, len(tn.inputData)/tn.batchSize+1)
	trace := make([]float64, tn.convergeCheck)

	allIdx := make([]int, len(tn.dataSet))
	for k:=0; k < len(allIdx); k++ {
		allIdx[k] = k
	}

	for i := 0; i < tn.numEpoch; i++ {
		tn.rng.Shuffle(len(allIdx), func(i, j int) { allIdx[i], allIdx[j] = allIdx[j], allIdx[i] })
		for j:=0; j<len(allIdx); j+=tn.batchSize {
			batch := int(math.Min(float64(len(allIdx)-j), float64(tn.batchSize)))
			inputs, truths := tn.SampleIdx(allIdx[j : j+batch])
			deriv[j/tn.batchSize], losses[j/tn.batchSize] = tn.Update(inputs, truths)
		}
		log.Printf("Training loss at step %d is %f, derivative is %f", i, ao.Average(losses), ao.Average(deriv))
		testLoss := tn.Test()
		trace[i % tn.convergeCheck] = testLoss
		if (i > tn.convergeCheck) && (trace[i % tn.convergeCheck] > trace[(i-1) % tn.convergeCheck]) && (trace[(i-1) % tn.convergeCheck] > trace[(i-2) % tn.convergeCheck]) {
			log.Println("Converged.")
			break
		}
	}
	return tn.nnFunc
}

func (tn *TransNetwork) CrossValidation() []network.Network {
	allData := make([][]float64, len(tn.dataSet))
	copy(allData, tn.dataSet)

	allIdx := make([]int, len(tn.dataSet))
	for k:=0; k < len(allIdx); k++ {
		allIdx[k] = k
	}
	tn.rng.Shuffle(len(allIdx), func(i, j int) { allIdx[i], allIdx[j] = allIdx[j], allIdx[i] })

	testSize := len(tn.dataSet) / 5
	var startTest float64
	var endTest float64
	endTests := make([]float64, 5)
	improves := make([]float64, 5)

	for j:=0; j < 5; j++ {
		// initialize network
		tn.NetworkInit()
		testIdx := make([]int, testSize)
		copy(testIdx, allIdx[j * testSize: (j+1) * testSize])

		var trainIdx []int
		for z:=0; z < (j * testSize); z++ {
			trainIdx = append(trainIdx, allIdx[z])
		}
		for z:=(j*(testSize+1)); z < len(allIdx); z++ {
			trainIdx = append(trainIdx, allIdx[z])
		}
		tn.dataSet = ao.SampleByIdx2d(allData, trainIdx)
		tn.testInputs, tn.testTruths = tn.organizeTest(testIdx, allData)

		log.Printf("Cross Validation %d.", j)
		startTest = tn.Test()
		tn.Train()
		endTest = tn.Test()
		endTests[j] = endTest
		improves[j] = endTest - startTest
	}
	log.Printf("End of cross validation. Averaged loss = %f. Averaged improvement = %f. \n", ao.Average(endTests), ao.Average(improves))
	return tn.nnFunc
}

func (tn *TransNetwork) SampleIdx(idx []int) ([][]float64, [][]float64) {
	batchSize := len(idx)
	inputs := make([][]float64, batchSize)
	truths := make([][]float64, batchSize)

	for i := range inputs {
		inputs[i] = make([]float64, tn.inputStateDim+tn.numAction)
		truths[i] = make([]float64, tn.outputStateDim+2)
	}
	for i := 0; i < batchSize; i++ {
		inputs[i] = tn.inputData[idx[i]]
		truths[i] = tn.groundTruth[idx[i]]
	}
	return inputs, truths
}

func (tn *TransNetwork) Update(inputs, truths [][]float64) (float64, float64){
	allLoss := make([]float64, len(tn.nnFunc))
	allDeriv := make([]float64, len(tn.nnFunc))
	var allTruth [][][]float64
	if len(tn.nnFunc) == 3 {
		allTruth = make([][][]float64, 3)
		allTruth[0] = ao.Index2d(truths, 0, len(truths), 0, tn.outputStateDim)
		allTruth[1] = ao.Index2d(truths, 0, len(truths), tn.outputStateDim, tn.outputStateDim+1)
		allTruth[2] = ao.Index2d(truths, 0, len(truths), tn.outputStateDim+1, tn.outputStateDim+2)
	} else {
		allTruth = make([][][]float64, 1)
		allTruth[0] = truths
	}
	for i:=0; i<len(tn.nnFunc); i++ {
		predicts := tn.nnFunc[i].Forward(inputs)
		allLoss[i] = loss.MseLoss(allTruth[i], predicts)
		deriv := loss.MseLossDeriv(allTruth[i], predicts)
		tn.nnFunc[i].Backward(deriv, tn.optimizer[i])
		allDeriv[i] = ao.Average(ao.Flatten2DFloat(deriv))
	}
	return ao.Average(allDeriv), ao.Average(allLoss)
}

func (tn *TransNetwork) Predict(state [][]float64, action []float64) ([][]float64){
	oneHotActs := ao.OneHotSet2D(action, tn.numAction)
	inputs := ao.Concatenate(state, oneHotActs)
	res := tn.PredictHelper(inputs)
	return res
}

func (tn *TransNetwork) PredictHelper(inputs [][]float64) ([][]float64){
	predicts := make([][][]float64, len(tn.nnFunc))
	for i:=0; i<len(tn.nnFunc); i++ {
		predicts[i] = tn.nnFunc[i].Predict(inputs)
	}
	var res [][]float64
	if tn.separatedNetwork {
		res = ao.Concatenate(ao.Concatenate(predicts[0], predicts[1]), predicts[2])
	} else {
		res = predicts[0]
	}
	return res
}

func (tn *TransNetwork) PredictSingleTrans(state []float64, action float64) ([]float64, float64, bool){
	tempS := make([][]float64, 1)
	tempA := make([]float64, 1)
	tempS[0] = state
	tempA[0] = action
	predicts := tn.Predict(tempS, tempA)
	states := ao.Index2d(predicts, 0, len(predicts), 0, tn.outputStateDim)[0]
	rewards := ao.Flatten2DFloat(ao.Index2d(predicts, 0, len(predicts), tn.outputStateDim, tn.outputStateDim+1))[0]
	tempT := ao.Flatten2DFloat(ao.Index2d(predicts, 0, len(predicts), tn.outputStateDim+1, tn.outputStateDim+2))[0]
	var terminal bool
	if tempT > 0.5 {
		terminal = true
	} else {
		terminal = false
	}
	return states, rewards, terminal
}


func (tn *TransNetwork) Test() (float64) {
	var inputs, truths [][]float64
	if tn.testInputs == nil {
		idxs := make([]int, 1000)
		for i:=0; i<len(idxs); i++ {
			idxs[i] = tn.rng.Intn(len(tn.inputData))
		}
		inputs, truths = tn.SampleIdx(idxs)
	} else {
		inputs = tn.testInputs
		truths = tn.testTruths
	}
	predicts := tn.PredictHelper(inputs)
	loss4Log := loss.MseLoss(truths, predicts)
	log.Printf("Test: Loss = %f", loss4Log)
	return loss4Log
}

func (tn *TransNetwork) organizeTest(idx []int, dataSet [][]float64) ([][]float64, [][]float64){
	batchSize := len(idx)
	inputs := make([][]float64, batchSize)
	truths := make([][]float64, batchSize)

	for i := range inputs {
		inputs[i] = make([]float64, tn.inputStateDim+tn.numAction)
		truths[i] = make([]float64, tn.outputStateDim+2)
	}
	for i := 0; i < batchSize; i++ {
		copy(inputs[i][:tn.inputStateDim], dataSet[idx[i]][:tn.inputStateDim])
		copy(inputs[i][tn.inputStateDim: ], ao.OneHotSet(dataSet[idx[i]][tn.inputStateDim], tn.numAction))
		copy(truths[i], dataSet[idx[i]][tn.inputStateDim+1:])
	}
	return inputs, truths
}
