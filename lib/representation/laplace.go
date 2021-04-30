package representation

import (
	"fmt"
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"github.com/stellentus/cartpoles/lib/util/network"
	"github.com/stellentus/cartpoles/lib/util/optimizer"
	"github.com/stellentus/cartpoles/lib/util/random"
	"log"
	"math"
	"math/rand"
)

type Laplace struct {
	seed	 		int
	rng 			*rand.Rand
	inputLen   		int
	hiddenLayer   	[]int
	repLen			int
	repFunc			network.Network
	optimizer		optimizer.Optimizer
	numStep			int
	beta        	float64
	delta 			float64
	batchSize   	int
	lambdaProb		[]float64
	trajLen			int
	learningRate   	float64
	dataSet 		[][][]float64
	//dataLenSet 		[]int
	testState 		[][]float64
	testClose 		[][]float64
	testFar 		[][]float64
	testForward 	int
}

func NewLaplace() *Laplace {
	return &Laplace{}
}

func (lp *Laplace) Initialize(seed int, numStep int, beta float64, delta float64, lambda float64, trajLen int,
	batchSize int, learningRate float64, hiddenLayer []int, dataSet [][]float64, terminSet []float64,
	inputLen int, repLen int, testForward int) {

	lp.seed = seed
	lp.rng = rand.New(rand.NewSource(int64(seed)))

	lp.numStep = numStep
	lp.beta = beta
	lp.delta = delta
	lp.batchSize = batchSize
	lp.learningRate = learningRate
	lp.trajLen = trajLen
	lp.testForward = testForward

	temp := make([]float64, trajLen)
	temp[0] = 0
	for i:=1; i<trajLen; i++ {
		temp[i] = math.Pow(lambda, float64(i))
	}
	lp.lambdaProb = temp

	//lp.dataSet, lp.dataLenSet = lp.OrganizeData(dataSet, terminSet)
	lp.dataSet = lp.OrganizeData(dataSet, terminSet)
	lp.testState = nil
	lp.inputLen = inputLen
	lp.hiddenLayer = hiddenLayer
	lp.repLen = repLen

	lp.NetworkInit()
}

func (lp *Laplace) NetworkInit() {
	lp.repFunc = network.CreateNetwork(lp.inputLen, lp.hiddenLayer, lp.repLen, lp.learningRate,
		0, 0, 0.9, 0.999, 1e-08)
	lp.optimizer = new(optimizer.Adam)
	lp.optimizer.Init(lp.learningRate, []float64{0.9, 0.999, 1e-08}, lp.inputLen, lp.hiddenLayer, lp.repLen)
}


func (lp *Laplace) OrganizeData(dataSet [][]float64, terminSet []float64) ([][][]float64) {
	var organized [][][]float64
	//var dataLen []int
	var termin bool
	var k int
	for i:=0; i<len(dataSet); i++ {
		var traj [][]float64
		termin = false
		k = 0
		for !termin && k < lp.trajLen && i+k < len(dataSet){
			traj = append(traj, dataSet[i+k])
			termin = (terminSet[i+k] == 1)
			k += 1
		}
		if k > 1 {
			organized = append(organized, traj)
			//dataLen = append(dataLen, k)
		}
		if len(traj) != k {
			fmt.Println("Length of traj should be same as k")
		}
	}
	//return organized, dataLen
	return organized
}

func (lp *Laplace) Train() network.Network {
	logTime := 1000
	deriv := make([]float64, logTime)
	losses := make([]float64, logTime)
	for i := 0; i < lp.numStep; i++ {
		states, closes, fars := lp.Sample(lp.batchSize)
		deriv[i%logTime], losses[i%logTime] = lp.Update(states, closes, fars)
		//if i%len(losses) == 0 && i!=0 {
		if i%len(losses) == 0 {
			//fmt.Println("Training loss at step", i, "is", ao.Average(losses), ". Derivative is", ao.Average(deriv))
			log.Printf("Training loss at step %d is %f, derivative is %f", i, ao.Average(losses), ao.Average(deriv))
			lp.Test(nil)
		}
	}
	return lp.repFunc
}

func (lp *Laplace) CrossValidation() network.Network {
	logTime := 1000
	deriv := make([]float64, logTime)
	losses := make([]float64, logTime)

	allData := make([][][]float64, len(lp.dataSet))
	//allDataLen := make([]int, len(lp.dataSet))
	copy(allData, lp.dataSet)
	//for t:=0; t<len(allData); t++ {
	//	fmt.Println(len(allData[t]), len(lp.dataSet[t]))
	//}
	//copy(allDataLen, lp.dataLenSet)

	allIdx := make([]int, len(lp.dataSet))
	for k:=0; k < len(allIdx); k++ {
		allIdx[k] = k
	}
	lp.rng.Shuffle(len(allIdx), func(i, j int) { allIdx[i], allIdx[j] = allIdx[j], allIdx[i] })

	testSize := len(lp.dataSet) / 5
	var startTest float64
	var endTest float64
	endTests := make([]float64, 5)
	improves := make([]float64, 5)

	for j:=0; j < 5; j++ {
		// initialize network
		lp.NetworkInit()
		testIdx := make([]int, testSize)
		copy(testIdx, allIdx[j * testSize: (j+1) * testSize])

		var trainIdx []int
		for z:=0; z < (j * testSize); z++ {
			trainIdx = append(trainIdx, allIdx[z])
		}
		for z:=(j*(testSize+1)); z < len(allIdx); z++ {
			trainIdx = append(trainIdx, allIdx[z])
		}
		lp.dataSet = ao.SampleByIdx3d(allData, trainIdx)
		//lp.dataLenSet = ao.SampleByIdx1dInt(allDataLen, trainIdx)
		//lp.testState, lp.testClose, lp.testFar = lp.organizeTest(testIdx, allData, allDataLen, lp.testForward)
		lp.testState, lp.testClose, lp.testFar = lp.organizeTest(testIdx, allData, lp.testForward)
		//for t:=0; t<len(lp.testState); t++ {
		//	//fmt.Println(len(lp.testState[t]), len(lp.testClose[t]), len(lp.testFar[t]))
		//	fmt.Println(lp.testState[t], lp.testClose[t], lp.testFar[t])
		//}

		log.Printf("Cross Validation %d.", j)
		startTest = lp.Test(testIdx)
		for i := 0; i < lp.numStep; i++ {
			states, closes, fars := lp.Sample(lp.batchSize)
			deriv[i%logTime], losses[i%logTime] = lp.Update(states, closes, fars)
			//if i%len(losses) == 0 && i!=0 {
			if i!=0 && i%len(losses) == 0 {
				//fmt.Println("Training loss at step", i, "is", ao.Average(losses), ". Derivative is", ao.Average(deriv))
				log.Printf("Step %d: Training loss = %f. Derivative = %f", i, ao.Average(losses), ao.Average(deriv))
				lp.Test(testIdx)
			}
		}
		endTest = lp.Test(testIdx)
		endTests[j] = endTest
		improves[j] = endTest - startTest
	}
	log.Printf("End of cross validation. Dynamic awareness = %f. Averaged improvement = %f. \n", ao.Average(endTests), ao.Average(improves))
	return lp.repFunc
}

func (lp *Laplace) Sample(batchSize int) ([][]float64, [][]float64, [][]float64){
	states := make([][]float64, batchSize)
	closes := make([][]float64, batchSize)
	fars := make([][]float64, batchSize)

	for i := range states {
		states[i] = make([]float64, lp.inputLen)
		closes[i] = make([]float64, lp.inputLen)
		fars[i] = make([]float64, lp.inputLen)
	}
	for i := 0; i < batchSize; i++ {
		chosen := lp.rng.Intn(len(lp.dataSet))
		normP := ao.NormalizeProb(lp.lambdaProb[:len(lp.dataSet[chosen])])
		//states[i] = lp.dataSet[chosen][len(lp.dataSet[chosen])-1-random.FreqSample(normP)]
		//closes[i] = lp.dataSet[chosen][len(lp.dataSet[chosen])-1]
		//randChosen := lp.rng.Intn(len(lp.dataSet))
		//fars[i] = lp.dataSet[randChosen][len(lp.dataSet[randChosen])-1]
		states[i] = lp.dataSet[chosen][random.FreqSample(normP)]
		closes[i] = lp.dataSet[chosen][0]
		randChosen := lp.rng.Intn(len(lp.dataSet))
		fars[i] = lp.dataSet[randChosen][0]
	}
	return states, closes, fars
}

func (lp *Laplace) organizeTest(idx []int, dataSet [][][]float64, forwardStep int) ([][]float64, [][]float64, [][]float64){
	batchSize := len(idx)
	states := make([][]float64, batchSize)
	closes := make([][]float64, batchSize)
	fars := make([][]float64, batchSize)

	for i := range states {
		states[i] = make([]float64, lp.inputLen)
		closes[i] = make([]float64, lp.inputLen)
		fars[i] = make([]float64, lp.inputLen)
	}
	for i := 0; i < batchSize; i++ {
		chosen := idx[i]
		if forwardStep==0 {
			normP := ao.NormalizeProb(lp.lambdaProb[:len(dataSet[chosen])])
			//states[i] = dataSet[chosen][len(dataSet[chosen])-1-random.FreqSample(normP)]
			states[i] = dataSet[chosen][random.FreqSample(normP)]
		} else {
			forwardTemp := int(math.Min(float64(len(dataSet[chosen])-1), float64(forwardStep)))
			//states[i] = dataSet[chosen][len(dataSet[chosen])-1-forwardStep]
			states[i] = dataSet[chosen][forwardTemp]
		}
		//closes[i] = dataSet[chosen][len(dataSet[chosen])-1]
		//randChosen := lp.rng.Intn(len(dataSet))
		//fars[i] = dataSet[randChosen][len(dataSet[randChosen])-1]
		closes[i] = dataSet[chosen][0]
		randChosen := lp.rng.Intn(len(dataSet))
		fars[i] = dataSet[randChosen][0]
	}
	return states, closes, fars
}


func (lp *Laplace) Update(states, closes, fars [][]float64) (float64, float64) {
	statesRep := lp.repFunc.Forward(states)
	closesRep := lp.repFunc.Forward(closes)
	farsRep := lp.repFunc.Forward(fars)
	//closesRep := lp.repFunc.Predict(closes)
	//farsRep := lp.repFunc.Predict(fars)

	attractiveLoss, alForLog := lp.GetAttractiveLoss(statesRep, closesRep)
	repulsiveLoss, rlForLog := lp.GetRepulsiveLoss(statesRep, farsRep)
	avgLoss := ao.BitwiseAdd(attractiveLoss, ao.A64ArrayMulti(lp.beta, repulsiveLoss))
	//fmt.Println(ao.Average(attractiveLoss), ao.Average(repulsiveLoss))
	statesAvgLossMat := make([][]float64, len(states))
	for i:=0; i<len(states); i++ {
		statesAvgLossMat[i] = make([]float64, lp.repLen)
		for j := 0; j < lp.repLen; j++ {
			statesAvgLossMat[i][j] = avgLoss[i]
		}
	}
	//lp.repFunc.Backward(statesAvgLossMat, lp.optimizer)

	closeLoss, clForLog := lp.GetAttractiveLoss(closesRep, statesRep)
	closeAvgLossMat := make([][]float64, len(states))
	for i:=0; i<len(states); i++ {
		closeAvgLossMat[i] = make([]float64, lp.repLen)
		for j := 0; j < lp.repLen; j++ {
			closeAvgLossMat[i][j] = closeLoss[i]
		}
	}
	//lp.repFunc.Forward(closes)
	//lp.repFunc.Backward(closeAvgLossMat, lp.optimizer)

	farLoss, fForLog := lp.GetRepulsiveLoss(farsRep, statesRep)
	farLoss = ao.A64ArrayMulti(lp.beta, farLoss)
	fForLog = ao.A64ArrayMulti(lp.beta, fForLog)
	farAvgLossMat := make([][]float64, len(states))
	for i:=0; i<len(states); i++ {
		farAvgLossMat[i] = make([]float64, lp.repLen)
		for j := 0; j < lp.repLen; j++ {
			farAvgLossMat[i][j] = farLoss[i]
		}
	}
	//lp.repFunc.Forward(fars)
	//lp.repFunc.Backward(farAvgLossMat, lp.optimizer)
	lp.repFunc.Backward(ao.BitwiseAdd2D(ao.BitwiseAdd2D(statesAvgLossMat, closeAvgLossMat), farAvgLossMat), lp.optimizer)
	//fmt.Println(ao.Average(alForLog), ao.Average(rlForLog), ao.Average(clForLog), ao.Average(fForLog))

	//return ao.Average(avgLoss)
	return ao.Average(ao.BitwiseAdd(ao.BitwiseAdd(ao.BitwiseAdd(attractiveLoss, repulsiveLoss), closeLoss), farLoss)),
		ao.Average(ao.BitwiseAdd(ao.BitwiseAdd(ao.BitwiseAdd(alForLog, rlForLog), clForLog), fForLog))
}

//func (lp *Laplace) GetAttractiveLoss(statesRep, closesRep [][]float64) [][]float64 {
func (lp *Laplace) GetAttractiveLoss(statesRep, closesRep [][]float64) ([]float64, []float64) {
	// For log
	temp := ao.BitwiseMinus2D(statesRep, closesRep)
	tempPow := ao.BitwisePower2D(temp, 2)
	beforeDer := ao.SumOnAxis2D(tempPow, 1)

	// For weight update
	diff := ao.BitwiseMinus2D(closesRep, statesRep)
	axisSum := ao.SumOnAxis2D(diff, 1)
	return axisSum, beforeDer
}

func (lp *Laplace) GetRepulsiveLoss(statesRep, farsRep [][]float64) ([]float64, []float64) {
	// For log
	//(\phi(u)^T \phi(v))^2
	tempdotProd := ao.SumOnAxis2D(ao.BitwisePower2D(ao.BitwiseMulti2D(statesRep, farsRep), 2), 1)
	// - delta*|\phi(u)|^2
	tempstatesNorm := ao.BitwisePower2D(statesRep, 2)
	tempstatesNormSum := ao.SumOnAxis2D(tempstatesNorm, 1)
	tempstatesNormWeighted := ao.A64ArrayMulti(lp.delta*(-1), tempstatesNormSum)
	// - delta*|\phi(v)|^2
	tempfarsNorm := ao.BitwisePower2D(farsRep, 2)
	tempfarsNormSum := ao.SumOnAxis2D(tempfarsNorm, 1)
	tempfarsNormWeighted := ao.A64ArrayMulti(lp.delta*(-1), tempfarsNormSum)
	//d := math.Pow(float64(lp.repLen) * lp.delta, 2)
	//repul := ao.Average(ao.BitwiseAdd(ao.BitwiseAdd(dotProd, statesNormWeighted), farsNormWeighted)) + d
	beforeDer := ao.BitwiseAdd(ao.BitwiseAdd(tempdotProd, tempstatesNormWeighted), tempfarsNormWeighted)
	//fmt.Println("Repulsive", ao.Average(beforeDer))

	// For weight update
	// d/du (\phi(u)^T \phi(v))^2
	dotProd := ao.SumOnAxis2D(ao.BitwiseMulti2D(ao.BitwiseMulti2D(statesRep, farsRep), farsRep), 1)
	// d/du (-|\phi(u)|^2)
	statesNormSum := ao.SumOnAxis2D(statesRep, 1)
	statesNormWeighted := ao.A64ArrayMulti(lp.delta*(-1), statesNormSum)

	//// d/du (-|\phi(v)|^2)
	//farsNormSum := ao.SumOnAxis2D(ao.Absolute2D(farsRep), 1)
	//farsNormWeighted := ao.A64ArrayMulti(lp.delta*(-1), farsNormSum)

	//d := math.Pow(float64(lp.repLen) * lp.delta, 2)
	//repul := ao.Average(ao.BitwiseAdd(ao.BitwiseAdd(dotProd, statesNormWeighted), farsNormWeighted)) + d
	//repul := ao.BitwiseAdd(ao.BitwiseAdd(dotProd, statesNormWeighted), farsNormWeighted)
	repul := ao.BitwiseAdd(dotProd, statesNormWeighted)

	return repul, beforeDer
}

func (lp *Laplace) Test(idx []int) float64 {
	var states, closes, fars [][]float64
	if lp.testState == nil {
		states, closes, fars = lp.Sample(1000)
	} else {
		states = lp.testState
		closes = lp.testClose
		fars = lp.testFar
	}
	statesRep := lp.repFunc.Predict(states)
	closesRep := lp.repFunc.Predict(closes)
	farsRep := lp.repFunc.Predict(fars)

	closeDist := ao.Average(ao.L2DistanceAxis1(statesRep, closesRep))
	farDist := ao.Average(ao.L2DistanceAxis1(statesRep, farsRep))
	da := (farDist - closeDist) / farDist
	log.Printf("Test: close pair distance = %f. far pair distance = %f. DynamicAwareness = %f", closeDist, farDist, da)
	return da
}
