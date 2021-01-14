package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"runtime"
	"sort"
	"time"

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
	seed          = flag.Uint64("seed", math.MaxUint64, "Seed to use; if 0xffffffffffffffff, use the time")
	numWorkers    = flag.Int("workers", -1, "Maximum number of workers; defaults to the number of CPUs if -1")
	numIterations = flag.Int("iterations", 3, "Total number of iterations")
	numSamples    = flag.Int("samples", 10, "Number of samples per iteration")
	numRuns       = flag.Int("runs", 2, "Number of runs per sample")
	numTimesteps  = flag.Int("timesteps", 1000, "Number of timesteps per run")
	percentElite  = flag.Float64("elite", 0.5, "Percent of samples that should be drawn from the elite group")
)

const e = 10.e-8

func main() {
	flag.Parse()

	if *numWorkers <= 0 {
		*numWorkers = runtime.NumCPU()
	}

	hyperparams := [5]string{"tilings", "tiles", "lambda", "epsilon", "adaptiveAlpha"}
	lower := [len(hyperparams)]float64{0.5, 0.5, 0.0, 0.0, 0.0}
	upper := [len(hyperparams)]float64{7.5, 4.5, 1.0, 1.0, 1}

	//typeOfHyperparams := [5]string{"discrete", "discrete", "continuous", "continuous", "continuous"}
	discreteHyperparamsIndices := [2]int64{0, 1}
	discreteRanges := [][]float64{[]float64{1, 2, 4, 8, 16, 32}, []float64{1, 2, 4}}
	discreteMidRanges := [][]float64{[]float64{1.5, 2.5, 3.5, 4.5, 5.5, 6.5}, []float64{1.5, 2.5, 3.5}}

	numElite := int64(float64(*numSamples) * *percentElite)
	numEliteElite := int(numElite / 2.0)

	var meanHyperparams [len(hyperparams)]float64
	for i := range meanHyperparams {
		meanHyperparams[i] = (lower[i] + upper[i]) / 2.0
	}

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

	covariance = nearestPD(covariance)

	symmetricCovariance := mat.NewSymDense(len(hyperparams), nil)
	for i := 0; i < len(hyperparams); i++ {
		for j := 0; j < len(hyperparams); j++ {
			symmetricCovariance.SetSym(i, j, (covariance.At(i, j)+covariance.At(j, i))/2.0)
		}
	}

	var choleskySymmetricCovariance mat.Cholesky
	choleskySymmetricCovariance.Factorize(symmetricCovariance)

	fmt.Println("Mean :", meanHyperparams)
	fmt.Println("")

	samples := make([][]float64, *numSamples)           //samples contain the original values of hyperparams (discrete, continuous)
	realvaluedSamples := make([][]float64, *numSamples) //realvaluedSamples contain the continuous representation of hyperparams (continuous)

	if *seed == math.MaxUint64 {
		*seed = uint64(time.Now().UnixNano())
	}
	rng := rand.New(rand.NewSource(*seed))
	i := 0

	for i < *numSamples {
		sample := distmv.NormalRand(nil, meanHyperparams[:], &choleskySymmetricCovariance, rand.NewSource(rng.Uint64()))
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

	// LOG THE MEAN OF THE DISTRIBUTION AFTER EVERY ITERATION

	for iteration := 0; iteration < *numIterations; iteration++ {
		fmt.Println("Iteration: ", iteration)
		fmt.Println("")
		samplesMetrics := make([]float64, *numSamples)
		fmt.Println("Samples before iteration: ", samples)
		fmt.Println("")

		jobs := make(chan int, *numSamples)
		results := make(chan averageAtIndex, *numSamples)

		for w := 0; w < *numWorkers; w++ {
			go worker(jobs, results, samples, *numRuns, iteration)
		}

		for s := 0; s < *numSamples; s++ {
			jobs <- s
		}
		close(jobs)

		count := 0
		for count < len(samples) {
			select {
			case avg := <-results:
				count++
				samplesMetrics[avg.idx] = avg.average
			}
		}

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

		elitePointsMatrix := mat.NewDense(len(elitePoints), len(hyperparams), nil)
		for rows := 0; rows < len(elitePoints); rows++ {
			for cols := 0; cols < len(hyperparams); cols++ {
				elitePointsMatrix.Set(rows, cols, elitePoints[rows][cols])
			}
		}

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

		for m := 0; m < int(numEliteElite); m++ {
			realvaluedSamples[m] = elitePoints[m]
			samples[m] = eliteSamplePoints[m]
		}
		i := int(numEliteElite)

		for i < *numSamples {
			sample := distmv.NormalRand(nil, meanHyperparams[:], &choleskySymmetricCovariance, rand.NewSource(rng.Uint64()))
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

func worker(jobs <-chan int, results chan<- averageAtIndex, samples [][]float64, numRuns, iteration int) {
	for idx := range jobs {
		average := runOneSample(samples[idx], numRuns, iteration)
		results <- averageAtIndex{
			average: average,
			idx:     idx,
		}
	}
}

func panicIfError(err error, reason string) {
	if err != nil {
		panic("ERROR " + err.Error() + ": " + reason)
	}
}

func nearestPD(A *mat.Dense) *mat.Dense {
	ARows, AColumns := A.Dims()

	B := mat.NewDense(ARows, AColumns, nil)

	transposedA := transpose(A)

	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			value := (A.At(i, j) + transposedA.At(i, j)) / 2.0
			B.Set(i, j, value)
		}
	}

	u, s, v, err := svd(B)

	if err != nil {
		fmt.Println(err)
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

	A2 := mat.NewDense(ARows, AColumns, nil)

	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			A2.Set(i, j, (B.At(i, j)+hDense.At(i, j))/2.0)
		}
	}

	A3 := mat.NewDense(ARows, AColumns, nil)

	transposedA2 := transpose(A2)

	for i := 0; i < ARows; i++ {
		for j := 0; j < AColumns; j++ {
			A3.Set(i, j, (A2.At(i, j)+transposedA2.At(i, j))/2.0)
		}
	}

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

func runOneSample(sample []float64, numRuns, iteration int) float64 {
	tilings := sample[0]
	tiles := sample[1]
	lambda := sample[2]
	epsilon := sample[3]
	adaptiveAlpha := sample[4]
	var run_metrics []float64
	for run := 0; run < numRuns; run++ {
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

		debug := logger.NewDebug(logger.DebugConfig{}) // TODO create a debug

		ag := &agent.ESarsa{Debug: debug}
		ag.InitializeWithSettings(agentSettings, lockweight.LockWeight{})

		env := &environment.Cartpole{Debug: debug}
		env.InitializeWithSettings(environment.CartpoleSettings{Seed: seed})

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
			MaxSteps:                *numTimesteps,
			DebugInterval:           0,
			DataPath:                "",
			ShouldLogTraces:         false,
			CacheTracesInRAM:        false,
			ShouldLogEpisodeLengths: false,
			MaxCPUs:                 1,
		}
		exp, err := experiment.New(ag, env, expConf, debug, data)
		panicIfError(err, "Couldn't create experiment")

		listOfRewards, _ := exp.Run()
		result := 0.0 // returns
		for index := 0; index < len(listOfRewards); index++ {
			result += listOfRewards[index]
		}

		run_metrics = append(run_metrics, result)
	}
	average := 0.0 //returns averaged across runs
	for _, v := range run_metrics {
		average += v
	}
	average /= float64(len(run_metrics))
	return average
}

type averageAtIndex struct {
	average float64
	idx     int
}
