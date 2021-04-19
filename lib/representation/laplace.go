package representation

import (
	"fmt"
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"github.com/stellentus/cartpoles/lib/util/network"
	"github.com/stellentus/cartpoles/lib/util/optimizer"
	"github.com/stellentus/cartpoles/lib/util/random"
	"math"
	"math/rand"
)

type Laplace struct {
	seed	 		int
	rng 			*rand.Rand
	inputLen   		int
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
	dataLenSet 		[]int
}

func NewLaplace() *Laplace {
	return &Laplace{}
}

func (lp *Laplace) Initialize(seed int, numStep int, beta float64, delta float64, lambda float64, trajLen int,
	batchSize int, learningRate float64, hiddenLayer []int, dataSet [][]float64, terminSet []float64,
	inputLen int, repLen int) {

	lp.seed = seed
	lp.rng = rand.New(rand.NewSource(int64(seed)))

	lp.numStep = numStep
	lp.beta = beta
	lp.delta = delta
	lp.batchSize = batchSize
	lp.learningRate = learningRate
	lp.trajLen = trajLen

	temp := make([]float64, trajLen)
	temp[0] = 0
	for i:=1; i<trajLen; i++ {
		temp[i] = math.Pow(lambda, float64(i))
	}
	lp.lambdaProb = temp

	lp.dataSet, lp.dataLenSet = lp.OrganizeData(dataSet, terminSet)
	lp.inputLen = inputLen
	lp.repLen = repLen

	lp.repFunc = network.CreateNetwork(lp.inputLen, hiddenLayer, lp.repLen, lp.learningRate,
		0, 0, 0.9, 0.999, 1e-08)
	lp.optimizer = new(optimizer.Adam)
	lp.optimizer.Init(lp.learningRate, []float64{0.9, 0.999, 1e-08}, lp.inputLen, hiddenLayer, lp.repLen)

}

func (lp *Laplace) OrganizeData(dataSet [][]float64, terminSet []float64) ([][][]float64, []int) {
	var organized [][][]float64
	var dataLen []int
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
			dataLen = append(dataLen, k)
		}
	}
	return organized, dataLen
}

func (lp *Laplace) Train() network.Network {
	losses := make([]float64, 1000)
	for i := 0; i < lp.numStep; i++ {
		states, closes, fars := lp.Sample(lp.batchSize)
		losses[i%len(losses)] = lp.Update(states, closes, fars)
		if i%len(losses) == 0 && i!=0 {
			fmt.Println("Training loss at step", i, "is", ao.Average(losses))
		}
	}
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
	for i := 0; i < lp.batchSize; i++ {
		//chosen := lp.rng.Intn(len(lp.dataSet))
		//states[i] = lp.dataSet[chosen][0]
		//
		//normP := ao.NormalizeProb(lp.lambdaProb[:lp.dataLenSet[chosen]])
		//closes[i] = lp.dataSet[chosen][random.FreqSample(normP)]
		//fars[i] = lp.dataSet[lp.rng.Intn(len(lp.dataSet))][0]

		chosen := lp.rng.Intn(len(lp.dataSet))
		normP := ao.NormalizeProb(lp.lambdaProb[:lp.dataLenSet[chosen]])
		states[i] = lp.dataSet[chosen][lp.dataLenSet[chosen]-1-random.FreqSample(normP)]
		closes[i] = lp.dataSet[chosen][lp.dataLenSet[chosen]-1]
		randChosen := lp.rng.Intn(len(lp.dataSet))
		fars[i] = lp.dataSet[randChosen][lp.dataLenSet[randChosen]-1]
	}
	return states, closes, fars
}

func (lp *Laplace) Update(states, closes, fars [][]float64) float64 {
	statesRep := lp.repFunc.Forward(states)
	closesRep := lp.repFunc.Predict(closes)
	farsRep := lp.repFunc.Predict(fars)

	attractiveLoss := lp.GetAttractiveLoss(statesRep, closesRep)
	repulsiveLoss := lp.GetRepulsiveLoss(statesRep, farsRep)
	avgLoss := ao.BitwiseAdd(attractiveLoss, ao.A64ArrayMulti(lp.beta, repulsiveLoss))
	//fmt.Println(ao.Average(attractiveLoss), ao.Average(repulsiveLoss))
	avgLossMat := make([][]float64, len(states))
	for i:=0; i<len(states); i++ {
		avgLossMat[i] = make([]float64, lp.repLen)
		for j := 0; j < lp.repLen; j++ {
			avgLossMat[i][j] = avgLoss[i]
		}
	}
	lp.repFunc.Backward(avgLossMat, lp.optimizer)

	//closeLoss := lp.GetAttractiveLoss(closesRep, statesRep)
	//avgLossMat = make([][]float64, len(states))
	//for i:=0; i<len(states); i++ {
	//	avgLossMat[i] = make([]float64, lp.repLen)
	//	for j := 0; j < lp.repLen; j++ {
	//		avgLossMat[i][j] = closeLoss[i]
	//	}
	//}
	//lp.repFunc.Forward(closes)
	//lp.repFunc.Backward(avgLossMat, lp.optimizer)
	//
	//farLoss := ao.A64ArrayMulti(lp.beta, lp.GetRepulsiveLoss(farsRep, statesRep))
	//avgLossMat = make([][]float64, len(states))
	//for i:=0; i<len(states); i++ {
	//	avgLossMat[i] = make([]float64, lp.repLen)
	//	for j := 0; j < lp.repLen; j++ {
	//		avgLossMat[i][j] = farLoss[i]
	//	}
	//}
	//lp.repFunc.Forward(fars)
	//lp.repFunc.Backward(avgLossMat, lp.optimizer)

	return ao.Average(avgLoss)
}

//func (lp *Laplace) GetAttractiveLoss(statesRep, closesRep [][]float64) [][]float64 {
func (lp *Laplace) GetAttractiveLoss(statesRep, closesRep [][]float64) []float64 {
	//diff := ao.BitwiseMinus2D(statesRep, closesRep)
	//diffPow := ao.BitwisePower2D(diff, 2)
	////diffPow = ao.A64ArrayMulti2D(0.5, diffPow)
	////return diffPow
	//axisSum := ao.SumOnAxis2D(diffPow, 1)
	////avg := ao.Average(axisSum) * 0.5
	////return avg
	//return axisSum

	diff := ao.BitwiseMinus2D(statesRep, closesRep)
	axisSum := ao.SumOnAxis2D(diff, 1)
	return axisSum
}

func (lp *Laplace) GetRepulsiveLoss(statesRep, farsRep [][]float64) []float64 {
	//// (\phi(u)^T \phi(v))^2
	//dotProd := ao.SumOnAxis2D(ao.BitwisePower2D(ao.BitwiseMulti2D(statesRep, farsRep), 2), 1)
	//
	//// - |\phi(u)|^2
	//statesNorm := ao.BitwisePower2D(statesRep, 2)
	//statesNormSum := ao.SumOnAxis2D(statesNorm, 1)
	//statesNormWeighted := ao.A64ArrayMulti(lp.delta*(-1), statesNormSum)
	//
	//// - |\phi(v)|^2
	//farsNorm := ao.BitwisePower2D(farsRep, 2)
	//farsNormSum := ao.SumOnAxis2D(farsNorm, 1)
	//farsNormWeighted := ao.A64ArrayMulti(lp.delta*(-1), farsNormSum)
	//
	////d := math.Pow(float64(lp.repLen) * lp.delta, 2)
	////repul := ao.Average(ao.BitwiseAdd(ao.BitwiseAdd(dotProd, statesNormWeighted), farsNormWeighted)) + d
	//repul := ao.BitwiseAdd(ao.BitwiseAdd(dotProd, statesNormWeighted), farsNormWeighted)
	//return repul

	// (\phi(u)^T \phi(v))^2
	dotProd := ao.SumOnAxis2D(ao.BitwiseMulti2D(ao.BitwiseMulti2D(statesRep, farsRep), farsRep), 1)

	// - |\phi(u)|^2
	statesNormSum := ao.SumOnAxis2D(statesRep, 1)
	statesNormWeighted := ao.A64ArrayMulti(lp.delta*(-1), statesNormSum)

	//// - |\phi(v)|^2
	//farsNormSum := ao.SumOnAxis2D(ao.Absolute2D(farsRep), 1)
	//farsNormWeighted := ao.A64ArrayMulti(lp.delta*(-1), farsNormSum)

	//d := math.Pow(float64(lp.repLen) * lp.delta, 2)
	//repul := ao.Average(ao.BitwiseAdd(ao.BitwiseAdd(dotProd, statesNormWeighted), farsNormWeighted)) + d
	//repul := ao.BitwiseAdd(ao.BitwiseAdd(dotProd, statesNormWeighted), farsNormWeighted)
	repul := ao.BitwiseAdd(dotProd, statesNormWeighted)

	return repul
}