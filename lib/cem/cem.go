package cem

import (
	"errors"
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

type Cem struct {
	// Seed is the random seed; if 0xffffffffffffffff, use the time
	Seed uint64

	// NumWorkers is the maximum number of workers
	// Defaults is the number of CPUs if -1
	NumWorkers int

	// NumIterations is the total number of iterations
	NumIterations int

	// NumSamples is the number of samples per iteration
	NumSamples int

	// NumRuns is the number of runs per sample
	NumRuns int

	// NumTimesteps is the number of timesteps per run
	NumTimesteps int

	// NumEpisodes is the number of episodes
	NumEpisodes int

	// NumStepsInEpisode is the number of steps in episode
	NumStepsInEpisode int

	// MaxRunLengthEpisodic is the max number of steps in episode
	MaxRunLengthEpisodic int

	// PercentElite is the percent of samples that should be drawn from the elite group
	PercentElite float64
}

const e = 10.e-8

func (cem Cem) Run() error {
	startTime := time.Now()

	if cem.NumWorkers <= 0 {
		cem.NumWorkers = runtime.NumCPU()
	}

	// Acrobot
	hyperparams := [5]string{"tilings", "tiles", "lambda", "wInit", "alpha"}
	lower := [len(hyperparams)]float64{0.5, 0.5, 0.0, -2.0, 0.0}
	upper := [len(hyperparams)]float64{4.5, 3.5, 1.0, 5.0, 1.0}

	discreteHyperparamsIndices := [2]int64{0, 1}
	discreteRanges := [][]float64{[]float64{8, 16, 32, 48}, []float64{2, 4, 8}}
	discreteMidRanges := [][]float64{[]float64{1.5, 2.5, 3.5, 4.5}, []float64{1.5, 2.5, 3.5}}

	numElite := int64(float64(cem.NumSamples) * cem.PercentElite)
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

	var err error
	if covariance, err = nearestPD(covariance); err != nil {
		return err
	}

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

	samples := make([][]float64, cem.NumSamples)           //samples contain the original values of hyperparams (discrete, continuous)
	realvaluedSamples := make([][]float64, cem.NumSamples) //realvaluedSamples contain the continuous representation of hyperparams (continuous)

	if cem.Seed == math.MaxUint64 {
		cem.Seed = uint64(time.Now().UnixNano())
	}
	rng := rand.New(rand.NewSource(cem.Seed))
	i := 0

	for i < cem.NumSamples {
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
						if sample[j] < discreteMidRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][k] {
							temp = append(temp, discreteRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][k])
							break
						}
					}
				}
			}
			samples[i] = temp
			i++
		}
	}

	// LOG THE MEAN OF THE DISTRIBUTION AFTER EVERY ITERATION

	for iteration := 0; iteration < cem.NumIterations; iteration++ {
		startIteration := time.Now()
		fmt.Println("Iteration: ", iteration)
		fmt.Println("")
		samplesMetrics := make([]float64, cem.NumSamples)
		fmt.Println("Samples before iteration: ", samples)
		fmt.Println("")

		jobs := make(chan int, cem.NumSamples)
		results := make(chan averageAtIndex, cem.NumSamples)

		for w := 0; w < cem.NumWorkers; w++ {
			go cem.worker(jobs, results, samples, cem.NumRuns, iteration)
		}

		for s := 0; s < cem.NumSamples; s++ {
			jobs <- s
		}
		close(jobs)

		count := 0
		for count < len(samples) {
			select {
			case avg := <-results:
				if avg.err != nil {
					err = avg.err // Only the most recent will be returned
				}
				count++
				samplesMetrics[avg.idx] = avg.average
			}
		}
		if err != nil {
			return err
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
		fmt.Println("Mean point: ", meanSampleHyperparams)

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

		if covariance, err = nearestPD(covWithE); err != nil {
			return err
		}

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

		for i < cem.NumSamples {
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
							if sample[j] < discreteMidRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][k] {
								temp = append(temp, discreteRanges[indexOfInt(int64(j), discreteHyperparamsIndices[:])][k])
								break
							}
						}
					}
				}
				samples[i] = temp
				i++
			}

		}
		fmt.Println("")
		fmt.Println("Execution time for iteration: ", time.Since(startIteration))
		fmt.Println("")
		fmt.Println("--------------------------------------------------")
	}
	fmt.Println("")
	fmt.Println("Execution time: ", time.Since(startTime))
	return nil
}

func (cem Cem) worker(jobs <-chan int, results chan<- averageAtIndex, samples [][]float64, numRuns, iteration int) {
	for idx := range jobs {
		average, err := cem.runOneSample(samples[idx], numRuns, iteration)
		results <- averageAtIndex{
			average: average,
			idx:     idx,
			err:     err,
		}
	}
}

func nearestPD(A *mat.Dense) (*mat.Dense, error) {
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
		return nil, err
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
		return A3, nil
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
			return nil, errors.New("Eigen decomposition failed")
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

	return A3, nil
}

func isPD(matrix mat.Matrix) bool {
	boolean := true

	var eig mat.Eigen
	ok := eig.Factorize(matrix, mat.EigenNone)
	if !ok {
		return false
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

func (cem Cem) runOneSample(sample []float64, numRuns, iteration int) (float64, error) {
	tilings := sample[0]
	tiles := sample[1]
	lambda := sample[2]
	wInit := sample[3]
	alpha := sample[4]
	var run_metrics []float64
	var run_successes []float64
	for run := 0; run < numRuns; run++ {
		seed := int64((numRuns * iteration) + run)
		agentSettings := agent.EsarsaSettings{
			EnableDebug:        false,
			Seed:               seed,
			NumTilings:         int(tilings),
			NumTiles:           int(tiles),
			Gamma:              1.0,
			Lambda:             float64(lambda),
			Epsilon:            0.0,
			Alpha:              float64(alpha),
			AdaptiveAlpha:      0.0,
			IsStepsizeAdaptive: false,
			WInit:              float64(wInit),
			EnvName:            "acrobot",
		}

		debug := logger.NewDebug(logger.DebugConfig{}) // TODO create a debug

		ag := &agent.ESarsa{Debug: debug}
		ag.InitializeWithSettings(agentSettings, lockweight.LockWeight{})

		env := &environment.Acrobot{Debug: debug}
		env.InitializeWithSettings(environment.AcrobotSettings{Seed: seed}) // Episodic acrobot

		// Does not log data yet
		data, err := logger.NewData(debug, logger.DataConfig{
			ShouldLogTraces:         false,
			CacheTracesInRAM:        false,
			ShouldLogEpisodeLengths: false,
			BasePath:                "", //"data/CEM"
			FileSuffix:              "", //strconv.Itoa(int(run))
		})
		if err != nil {
			return 0, err
		}

		expConf := config.Experiment{
			MaxEpisodes:             50000,
			MaxRunLengthEpisodic:    cem.MaxRunLengthEpisodic,
			DebugInterval:           0,
			DataPath:                "",
			ShouldLogTraces:         false,
			CacheTracesInRAM:        false,
			ShouldLogEpisodeLengths: false,
			MaxCPUs:                 1,
		}
		exp, err := experiment.New(ag, env, expConf, debug, data)
		if err != nil {
			return 0, err
		}

		listOfListOfRewards, _ := exp.Run()
		var listOfRewards []float64

		//Episodic Acrobot, last 1/10th of the episodes
		for i := 0; i < len(listOfListOfRewards); i++ {
			for j := 0; j < len(listOfListOfRewards[i]); j++ {
				listOfRewards = append(listOfRewards, listOfListOfRewards[i][j])
			}
		}

		result := len(listOfRewards)
		successes := len(listOfListOfRewards)

		run_metrics = append(run_metrics, float64(result))
		run_successes = append(run_successes, float64(successes))
	}
	average := 0.0 //returns averaged across runs
	average_success := 0.0
	for _, v := range run_metrics {
		average += v
	}
	for _, v := range run_successes {
		average_success += v
	}
	average /= float64(len(run_metrics))
	average_success /= float64(len(run_successes))
	average_steps_to_failure := (average) / (average_success)
	return -average_steps_to_failure, nil //episodic  acrobot, returns negative of steps to failure
}

type averageAtIndex struct {
	average float64
	idx     int
	err     error
}
