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
	stateDim  		int
	hiddenLayer   	[]int
	nnFunc			network.Network
	optimizer		optimizer.Optimizer
	learningRate   	float64
	batchSize   	int
	numEpoch		int

	inputData		[][]float64
	groundTruth		[][]float64

	dataSet 		[][]float64
	testInputs 		[][]float64
	testTruths 		[][]float64

	convergeCheck	int
}

func New() *TransNetwork {
	return &TransNetwork{}
}

func (tn *TransNetwork) Initialize(seed int64, dataSet [][]float64, numEpoch int, batchSize int, learningRate float64, hiddenLayer []int,
	stateDim int, numAction int) {

	tn.seed = seed
	tn.rng = rand.New(rand.NewSource(int64(seed)))
	tn.dataSet = dataSet
	tn.numEpoch = numEpoch
	tn.stateDim = stateDim
	tn.numAction = numAction
	tn.hiddenLayer = hiddenLayer
	tn.learningRate = learningRate
	tn.batchSize = batchSize
	tn.NetworkInit()
	tn.convergeCheck = 3
}

func (tn *TransNetwork) NetworkInit() {
	inputLen := tn.stateDim+tn.numAction
	outputLen := tn.stateDim+2
	tn.nnFunc = network.CreateNetwork(inputLen, tn.hiddenLayer, outputLen, tn.learningRate,
		0, 0, 0.9, 0.999, 1e-08)
	tn.optimizer = new(optimizer.Adam)
	tn.optimizer.Init(tn.learningRate, []float64{0.9, 0.999, 1e-08}, inputLen, tn.hiddenLayer, outputLen)
}

func (tn *TransNetwork) OrganizeData(dataSet [][]float64) ([][]float64, [][]float64) {
	input := make([][]float64, len(dataSet))
	output := make([][]float64, len(dataSet))

	for i:=0; i<len(dataSet); i++ {
		input[i] = make([]float64, tn.stateDim+tn.numAction)
		output[i] = make([]float64, tn.stateDim+2)
		action := dataSet[i][tn.stateDim]
		oneHotA := ao.OneHotSet(action, tn.numAction)
		//fmt.Println(dataSet[i][: tn.stateDim])
		copy(input[i][: tn.stateDim], dataSet[i][: tn.stateDim])
		copy(input[i][tn.stateDim: ], oneHotA)
		copy(output[i], dataSet[i][tn.stateDim+1: ])
	}

	return input, output
}

func (tn *TransNetwork) Train() network.Network {
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

func (tn *TransNetwork) CrossValidation() network.Network {
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
		inputs[i] = make([]float64, tn.stateDim+tn.numAction)
		truths[i] = make([]float64, tn.stateDim+2)
	}
	for i := 0; i < batchSize; i++ {
		inputs[i] = tn.inputData[idx[i]]
		truths[i] = tn.groundTruth[idx[i]]
	}
	return inputs, truths
}

func (tn *TransNetwork) Update(inputs, truths [][]float64) (float64, float64){
	predicts := tn.nnFunc.Forward(inputs)
	loss4Log := loss.MseLoss(truths, predicts)
	deriv := loss.MseLossDeriv(truths, predicts)
	tn.nnFunc.Backward(deriv, tn.optimizer)
	return ao.SumOnAxis2D(deriv, 0)[0] / float64(len(deriv)*len(deriv[0])), loss4Log
}

func (tn *TransNetwork) Predict(state [][]float64, action []float64) ([][]float64){
	oneHotActs := ao.OneHotSet2D(action, tn.numAction)
	inputs := ao.Concatenate(state, oneHotActs)
	predicts := tn.nnFunc.Predict(inputs)
	return predicts
}

func (tn *TransNetwork) PredictSingleTrans(state []float64, action float64) ([]float64, float64, bool){
	tempS := make([][]float64, 1)
	tempA := make([]float64, 1)
	tempS[0] = state
	tempA[0] = action
	predicts := tn.Predict(tempS, tempA)
	states := ao.Index2d(predicts, 0, len(predicts), 0, tn.stateDim)[0]
	rewards := ao.Flatten2DFloat(ao.Index2d(predicts, 0, len(predicts), tn.stateDim, tn.stateDim+1))[0]
	tempT := ao.Flatten2DFloat(ao.Index2d(predicts, 0, len(predicts), tn.stateDim+1, tn.stateDim+2))[0]
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

	predicts := tn.nnFunc.Predict(inputs)
	loss4Log := loss.MseLoss(truths, predicts)

	log.Printf("Test: Loss = %f", loss4Log)
	return loss4Log
}

func (tn *TransNetwork) organizeTest(idx []int, dataSet [][]float64) ([][]float64, [][]float64){
	batchSize := len(idx)
	inputs := make([][]float64, batchSize)
	truths := make([][]float64, batchSize)

	for i := range inputs {
		inputs[i] = make([]float64, tn.stateDim+tn.numAction)
		truths[i] = make([]float64, tn.stateDim+2)
	}
	for i := 0; i < batchSize; i++ {
		copy(inputs[i][:tn.stateDim], dataSet[idx[i]][:tn.stateDim])
		copy(inputs[i][tn.stateDim: ], ao.OneHotSet(dataSet[idx[i]][tn.stateDim], tn.numAction))
		copy(truths[i], dataSet[idx[i]][tn.stateDim+1:])
	}
	return inputs, truths
}
