package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"runtime"
	"sort"

	"github.com/mkmik/argsort"
	"github.com/stellentus/cartpoles/lib/agent"
	"github.com/stellentus/cartpoles/lib/config"
	"github.com/stellentus/cartpoles/lib/environment"
	"github.com/stellentus/cartpoles/lib/experiment"
	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/util/lockweight"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distmv"
)

var (
	cpus = flag.Int("cpus", 2, "Number of CPUs")
)

func main() {
	flag.Parse()

	runtime.GOMAXPROCS(*cpus) // Limit the number of CPUs to the provided value (unchanged if the input is <1)

	debug := logger.NewDebug(logger.DebugConfig{}) // TODO create a debug

	numTimesteps := 75000
	//gamma := 0.9
	hyperparams := [5]string{"tilings", "tiles", "lambda", "epsilon", "adaptiveAlpha"}
	lower := [len(hyperparams)]float64{0.5, 0.5, 0.0, 0.0, 0.0}
	upper := [len(hyperparams)]float64{7.5, 4.5, 1.0, 1.0, 1}

	//typeOfHyperparams := [5]string{"discrete", "discrete", "continuous", "continuous", "continuous"}
	discreteHyperparamsIndices := [2]int64{0, 1}
	discreteRanges := [][]float64{[]float64{1, 2, 4, 8, 16, 32}, []float64{1, 2, 4}}
	discreteMidRanges := [][]float64{[]float64{1.5, 2.5, 3.5, 4.5, 5.5, 6.5}, []float64{1.5, 2.5, 3.5}}

	numSamples := 20 // 300
	percentElite := 0.5
	numElite := int64(float64(numSamples) * percentElite)
	e := math.Pow(10, -8)
	numIterations := 10
	numRuns := 1

	var meanHyperparams [len(hyperparams)]float64
	for i := range meanHyperparams {
		meanHyperparams[i] = (lower[i] + upper[i]) / 2.0
	}

	//covariance := make([][]float64, len(hyperparams))
	//for i := range covariance {
	//	covariance[i] = make([]float64, len(hyperparams))
	//}

	// Copy all of this
	covariance := mat.NewDense(len(hyperparams), len(hyperparams), nil)
	covarianceRows, covarianceColumns := covariance.Dims()

	minLower := lower[0]
	maxUpper := upper[0]

	for _, v := range lower {
		if v < minLower {
			minLower = v
		}
	}
	for _, v := range upper {
		if v > maxUpper {
			maxUpper = v
		}
	}

	for i := 0; i < covarianceRows; i++ {
		for j := 0; j < covarianceColumns; j++ {
			if i == j {
				covariance.Set(i, j, math.Pow(maxUpper-minLower, 2)+e)
			} else {
				covariance.Set(i, j, e)
			}
		}
	}

	//fmt.Println("Before")
	//matPrint(covariance)
	covariance = nearestPD(covariance)
	//fmt.Println("After")
	//matPrint(covariance)

	symmetricCovariance := mat.NewSymDense(len(hyperparams), nil)
	for i := 0; i < len(hyperparams); i++ {
		for j := 0; j < len(hyperparams); j++ {
			symmetricCovariance.SetSym(i, j, (covariance.At(i, j)+covariance.At(j, i))/2.0)
		}
	}

	//fmt.Println("Symmetric")
	//matPrint(symmetricCovariance)

	var choleskySymmetricCovariance mat.Cholesky
	choleskySymmetricCovariance.Factorize(symmetricCovariance)

	fmt.Println("Mean :", meanHyperparams)
	fmt.Println("")
	//fmt.Println(symmetricCovariance)

	samples := make([][]float64, numSamples)
	realvaluedSamples := make([][]float64, numSamples)

	//Think of how to make this random
	var src rand.Source
	i := 0

	for i < numSamples {
		sample := distmv.NormalRand(nil, meanHyperparams[:], &choleskySymmetricCovariance, src)
		flag := 0
		for j := 0; j < len(hyperparams); j++ {
			if sample[j] < lower[j] || sample[j] > upper[j] {
				flag = 1
				break
			}
		}
		if flag == 0 {
			realvaluedSamples[i] = sample
			var temp []float64
			for j := 0; j < len(hyperparams); j++ {
				if !containsInt(discreteHyperparamsIndices[:], int64(j)) {
					temp = append(temp, sample[j])
				} else {
					for k := 0; k < len(discreteMidRanges[j]); k++ {
						if sample[j] <= discreteMidRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][k] {
							temp = append(temp, discreteRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][k])
							break
						}
					}
					if sample[j] > discreteMidRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][len(discreteMidRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])])-1] {
						temp = append(temp, discreteRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][len(discreteRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])])-1])
					}

				}
			}
			samples[i] = temp
			i++
		}

	}

	fmt.Println("Samples: ", samples)
	fmt.Println("")

	// to create MVND you need to use gonum NewNormal and covariance = (covariance + covariance.T)/2.0 to make it symmetric
	//for i := 0; i < numSamples; i++ {
	//fmt.Println(realvaluedSamples[i])
	//fmt.Println(samples[i])
	//fmt.Println("")
	//fmt.Println("")
	//}

	// LOG THE MEAN OF THE DISTRIBUTION AFTER EVERY ITERATION

	//fmt.Println("-------------------")
	//fmt.Println("-------------------")
	//fmt.Println(numTimesteps, numRuns, hyperparams, lower, upper, discreteHyperparamsIndices, discreteMidRanges, discreteRanges, numSamples, numElite, e, numIterations, numRuns, meanHyperparams, covariance)

	//iterations
	//    samples
	//        runs

	for iteration := 0; iteration < numIterations; iteration++ {
		fmt.Println("Iteration: ", iteration)
		fmt.Println("")
		var samplesMetrics []float64
		fmt.Println("Samples before iteration: ", samples)
		fmt.Println("")
		for s := 0; s < len(samples); s++ {
			//fmt.Println("Sample number: ", s)
			//fmt.Println("Sample: ", samples[s])
			//fmt.Println("Real valued: ", realvaluedSamples[s])
			tilings := samples[s][0]
			tiles := samples[s][1]
			lambda := samples[s][2]
			epsilon := samples[s][3]
			adaptiveAlpha := samples[s][4]
			var run_metrics []float64
			for run := 0; run < numRuns; run++ {
				//agentSettings := agent.DefaultESarsaSettings()
				// Do CEM stuff to change settings and SEED
				seed := int64((numRuns * iteration) + run)
				agentSettings := agent.EsarsaSettings{
					EnableDebug:        false,
					Seed:               seed,
					NumTilings:         int(tilings),
					NumTiles:           int(tiles),
					Gamma:              0.9,
					Lambda:             float64(lambda),
					Epsilon:            float64(epsilon),
					Alpha:              0.0,
					AdaptiveAlpha:      float64(adaptiveAlpha),
					IsStepsizeAdaptive: true,
				}
				//fmt.Println("Agent Settings: ", agentSettings)

				ag := &agent.ESarsa{Debug: debug}
				ag.InitializeWithSettings(agentSettings, lockweight.LockWeight{})

				env := &environment.Cartpole{Debug: debug}
				env.InitializeWithSettings(environment.CartpoleSettings{Seed: seed}) // TODO change seed

				// Does not log data yet
				data, err := logger.NewData(debug, logger.DataConfig{
					ShouldLogTraces:         false,
					CacheTracesInRAM:        false,
					ShouldLogEpisodeLengths: false,
					BasePath:                "", //"data/CEM"
					FileSuffix:              "", //strconv.Itoa(int(run))
				})
				panicIfError(err, "Couldn't create logger.Data")

				expConf := config.Experiment{
					MaxEpisodes:             0,
					MaxSteps:                numTimesteps,
					DebugInterval:           0,
					DataPath:                "",
					ShouldLogTraces:         false,
					CacheTracesInRAM:        false,
					ShouldLogEpisodeLengths: false,
					MaxCPUs:                 *cpus,
				}
				exp, err := experiment.New(ag, env, expConf, debug, data)
				panicIfError(err, "Couldn't create experiment")

				listOfRewards, _ := exp.Run()
				result := 0.0
				for index := 0; index < len(listOfRewards); index++ {
					result += listOfRewards[index]
				}

				run_metrics = append(run_metrics, result)
				//fmt.Println("Iteration: ", iteration)
				//fmt.Println("Sample: ", s)
				//fmt.Println("Run: ", run)
				//fmt.Println(result)

				//fmt.Println(data.NumberOfEpisodes())
			}
			//fmt.Println(run_metrics)
			average := 0.0
			for _, v := range run_metrics {
				average += v
			}
			average /= float64(len(run_metrics))
			//fmt.Println("Performance: ", average)
			//fmt.Println("")
			samplesMetrics = append(samplesMetrics, average)

		}
		//fmt.Println(realvaluedSamples)
		//fmt.Println("")
		fmt.Println("Sample Metric: ", samplesMetrics)
		fmt.Println("")
		ascendingIndices := argsort.Sort(sort.Float64Slice(samplesMetrics))
		descendingIndices := make([]int, len(samples))
		for ind := 0; ind < len(samples); ind++ {
			descendingIndices[len(samples)-1-ind] = ascendingIndices[ind]
		}

		descendingSamplesMetrics := make([]float64, len(samples))
		descendingSamples := make([][]float64, len(samples))
		descendingRealValuedSamples := make([][]float64, len(samples))
		for ds := 0; ds < len(samples); ds++ {
			descendingSamplesMetrics[ds] = samplesMetrics[descendingIndices[ds]]
			descendingSamples[ds] = samples[descendingIndices[ds]]
			descendingRealValuedSamples[ds] = realvaluedSamples[descendingIndices[ds]]
		}

		elitePoints := make([][]float64, numElite)
		eliteSamplePoints := make([][]float64, numElite)
		elitePoints = descendingRealValuedSamples[:numElite]
		eliteSamplePoints = descendingSamples[:numElite]
		var meanSampleHyperparams [len(hyperparams)]float64
		copy(meanHyperparams[:], elitePoints[0])
		copy(meanSampleHyperparams[:], eliteSamplePoints[0])
		fmt.Println("Elite points: ", eliteSamplePoints)
		fmt.Println("")
		fmt.Println("Elite Points Metric: ", descendingSamplesMetrics[:numElite])
		fmt.Println("")
		//fmt.Println("--------------------------------------------")

		elitePointsMatrix := mat.NewDense(len(elitePoints), len(hyperparams), nil)
		for rows := 0; rows < len(elitePoints); rows++ {
			for cols := 0; cols < len(hyperparams); cols++ {
				elitePointsMatrix.Set(rows, cols, elitePoints[rows][cols])
			}
		}
		//fmt.Println(elitePoints)
		//matPrint(elitePointsMatrix)

		cov := mat.NewSymDense(len(hyperparams), nil)
		stat.CovarianceMatrix(cov, elitePointsMatrix, nil)

		covWithE := mat.NewDense(len(hyperparams), len(hyperparams), nil)
		for rows := 0; rows < len(hyperparams); rows++ {
			for cols := 0; cols < len(hyperparams); cols++ {
				if rows == cols {
					covWithE.Set(rows, cols, cov.At(rows, cols)+e)
				} else {
					covWithE.Set(rows, cols, cov.At(rows, cols)-e)
				}
			}
		}

		covariance = nearestPD(covWithE)

		symmetricCovariance := mat.NewSymDense(len(hyperparams), nil)
		for i := 0; i < len(hyperparams); i++ {
			for j := 0; j < len(hyperparams); j++ {
				symmetricCovariance.SetSym(i, j, (covariance.At(i, j)+covariance.At(j, i))/2.0)
			}
		}

		var choleskySymmetricCovariance mat.Cholesky
		choleskySymmetricCovariance.Factorize(symmetricCovariance)

		//samples := make([][]float64, numSamples)
		//realvaluedSamples := make([][]float64, numSamples)

		for m := 0; m < int(numElite/2); m++ {
			realvaluedSamples[m] = elitePoints[m]
			samples[m] = eliteSamplePoints[m]
		}
		i := int(numElite / 2.0)

		//fmt.Println(meanHyperparams)
		//fmt.Println(symmetricCovariance)
		for i < numSamples {
			sample := distmv.NormalRand(nil, meanHyperparams[:], &choleskySymmetricCovariance, src)
			flag := 0
			for j := 0; j < len(hyperparams); j++ {
				if sample[j] < lower[j] || sample[j] > upper[j] {
					flag = 1
					break
				}
			}
			if flag == 0 {
				realvaluedSamples[i] = sample
				var temp []float64
				for j := 0; j < len(hyperparams); j++ {
					if !containsInt(discreteHyperparamsIndices[:], int64(j)) {
						temp = append(temp, sample[j])
					} else {
						for k := 0; k < len(discreteMidRanges[j]); k++ {
							if sample[j] <= discreteMidRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][k] {
								temp = append(temp, discreteRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][k])
								break
							}
						}
						if sample[j] > discreteMidRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][len(discreteMidRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])])-1] {
							temp = append(temp, discreteRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][len(discreteRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])])-1])
						}

					}
				}
				samples[i] = temp
				i++
			}

		}

		fmt.Println("--------------------------------------------------")
	}
}

func panicIfError(err error, reason string) {
	if err != nil {
		panic("ERROR " + err.Error() + ": " + reason)
	}
}

func nearestPD(A *mat.Dense) *mat.Dense {
	ARows, AColumns := A.Dims()
	//fmt.Println("A")
	//matPrint(A)
	//fmt.Println("B")

	B := mat.NewDense(ARows, AColumns, nil)

	transposedA := transpose(A)

	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			value := (A.At(i, j) + transposedA.At(i, j)) / 2.0
			B.Set(i, j, value)
		}
	}
	//matPrint(B)

	u, s, v, err := svd(B)
	//fmt.Println("U")
	//matPrint(&u)
	//fmt.Println("s")
	//fmt.Println(s)
	//fmt.Println("V")
	//matPrint(&v)

	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println("")
		//fmt.Println(u)
		//fmt.Println(s)
		//fmt.Println(v)

	}

	uRows, uColumns := u.Dims()
	vRows, vColumns := v.Dims()
	uDense := mat.NewDense(uRows, uColumns, nil)
	vDense := mat.NewDense(vRows, vColumns, nil)

	for i := 0; i < uRows; i++ {
		for j := 0; j < uColumns; j++ {
			uDense.Set(i, j, u.At(i, j))
		}
	}

	for i := 0; i < vRows; i++ {
		for j := 0; j < vColumns; j++ {
			vDense.Set(i, j, v.At(i, j))
		}
	}

	diagonalSMatrix := mat.NewDense(ARows, AColumns, nil)
	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			if i == j {
				diagonalSMatrix.Set(i, j, s[i])
			} else {
				diagonalSMatrix.Set(i, j, 0.0)
			}

		}
	}

	var temp mat.Dense
	var original mat.Dense
	temp.Mul(uDense, diagonalSMatrix)
	original.Mul(&temp, transpose(vDense))
	originalRows, originalColumns := original.Dims()
	originalDense := mat.NewDense(originalRows, originalColumns, nil)

	for i := 0; i < originalRows; i++ {
		for j := 0; j < originalColumns; j++ {
			originalDense.Set(i, j, original.At(i, j))
		}
	}
	//fmt.Println("Original")
	//matPrint(originalDense)

	var temp0 mat.Dense
	var H mat.Dense
	temp0.Mul(diagonalSMatrix, vDense)
	H.Mul(transpose(vDense), &temp0)
	hRows, hColumns := H.Dims()
	hDense := mat.NewDense(hRows, hColumns, nil)

	for i := 0; i < hRows; i++ {
		for j := 0; j < hColumns; j++ {
			hDense.Set(i, j, H.At(i, j))
		}
	}

	//fmt.Println("H")
	//matPrint(hDense)

	A2 := mat.NewDense(ARows, AColumns, nil)

	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			A2.Set(i, j, (B.At(i, j)+hDense.At(i, j))/2.0) // it is H[i][j]
		}
	}

	//fmt.Println("A2")
	//matPrint(A2)
	A3 := mat.NewDense(ARows, AColumns, nil)

	transposedA2 := transpose(A2)

	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			A3.Set(i, j, (A2.At(i, j)+transposedA2.At(i, j))/2.0)
		}
	}

	//fmt.Println("A3")
	//matPrint(A3)
	//fmt.Println(A3)

	if isPD(A3) {
		return A3
	}

	normA := mat.Norm(A, 2)
	nextNearestDistance := math.Nextafter(normA, normA+1) - normA
	previousNearestDistance := normA - math.Nextafter(normA, normA-1)

	spacing := math.Min(nextNearestDistance, previousNearestDistance)

	I := mat.NewDense(ARows, ARows, nil)
	for i := 0; i < ARows; i++ {
		for j := 0; j < ARows; j++ {
			if i == j {
				I.Set(i, j, 1)
			} else {
				I.Set(i, j, 0)
			}
		}
	}

	k := 1.0

	for !isPD(A3) {

		var eig mat.Eigen
		ok := eig.Factorize(A3, mat.EigenNone)
		if !ok {
			fmt.Println("Eigen decomposition failed")
		}
		eigenvalues := eig.Values(nil)

		realeigenvalues := make([]float64, len(eigenvalues))
		for i := range eigenvalues {
			realeigenvalues[i] = real(eigenvalues[i])
		}

		minrealeigenvalues := realeigenvalues[0]
		for _, value := range realeigenvalues {
			if minrealeigenvalues > value {
				minrealeigenvalues = value
			}
		}
		for i := 0; i < ARows; i++ {
			for j := 0; j < AColumns; j++ {
				A3.Set(i, j, I.At(i, j)*((-minrealeigenvalues*math.Pow(k, 2))+spacing))
			}
		}
		k++
	}

	return A3
}

func isPD(matrix mat.Matrix) bool {
	boolean := true

	var eig mat.Eigen
	ok := eig.Factorize(matrix, mat.EigenNone)
	if !ok {
		fmt.Println("Eigen decomposition failed")
	}
	eigenvalues := eig.Values(nil)

	realeigenvalues := make([]float64, len(eigenvalues))
	for i := range eigenvalues {
		realeigenvalues[i] = real(eigenvalues[i])
	}

	for i := range realeigenvalues {
		if realeigenvalues[i] <= 0 {
			boolean = false
			break
		}
	}

	return boolean
}

func transpose(matrix *mat.Dense) *mat.Dense {
	rows, columns := matrix.Dims()
	transposeMatrix := mat.NewDense(rows, columns, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			transposeMatrix.Set(i, j, matrix.At(j, i))
		}
	}
	return transposeMatrix
}

func svd(Matrix mat.Matrix) (mat.Dense, []float64, mat.Dense, error) {
	var svd mat.SVD
	if ok := svd.Factorize(Matrix, mat.SVDFull); !ok {
		var nilMat mat.Dense
		return nilMat, nil, nilMat, errors.New("SVD factorization failed")
	}
	var v, u mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)
	s := svd.Values(nil)
	return u, s, v, nil
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func containsInt(s []int64, e int64) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}

// Works if data has unique elements without repetition
func indexOfInt(element int64, data []int64) int {
	for k, v := range data {
		if element == v {
			return k
		}
	}
	return -1 //not found.
}
