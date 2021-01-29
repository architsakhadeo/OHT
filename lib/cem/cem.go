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
	getSets AgentSettingsProvider

	// numWorkers is the maximum number of workers
	// Defaults is the number of CPUs if -1
	numWorkers int

	// numIterations is the total number of iterations
	numIterations int

	// numSamples is the number of samples per iteration
	numSamples int

	// numRuns is the number of runs per sample
	numRuns int

	// numTimesteps is the number of timesteps per run
	numTimesteps int

	// numEpisodes is the number of episodes
	numEpisodes int

	// numStepsInEpisode is the number of steps in episode
	numStepsInEpisode int

	// maxRunLengthEpisodic is the max number of steps in episode
	maxRunLengthEpisodic int

	// percentElite is the percent of samples that should be drawn from the elite group
	percentElite float64

	debug logger.Debug
	data  logger.Data

	rng *rand.Rand

	discreteHyperparamsIndices []int64
	hyperparams                []string
	discreteRanges             [][]float64
	discreteMidRanges          [][]float64
	numElite                   int64
	numEliteElite              int
	meanHyperparams            []float64
	symmetricCovariance        *mat.SymDense
	lower                      []float64
	upper                      []float64
}

// AgentSettingsProvider is a function that returns agent settings corresponding to the provided seed and slice of hyperparameters.
type AgentSettingsProvider func(seed int64, hyperparameters []float64) agent.EsarsaSettings

const e = 10.e-8

func New(getSets AgentSettingsProvider, opts ...Option) (*Cem, error) {
	if getSets == nil {
		return nil, errors.New("Cem requires a settings provider")
	}
	// Initialize with default values
	cem := &Cem{
		getSets:                    getSets,
		numWorkers:                 runtime.NumCPU(),
		numIterations:              3,
		numSamples:                 10,
		numRuns:                    2,
		numEpisodes:                -1,
		numStepsInEpisode:          -1,
		percentElite:               0.5,
		debug:                      logger.NewDebug(logger.DebugConfig{}),
		discreteHyperparamsIndices: []int64{0, 1},
		hyperparams:                []string{"tilings", "tiles", "lambda", "wInit", "alpha"},
		discreteRanges:             [][]float64{[]float64{8, 16, 32, 48}, []float64{2, 4, 8}},
		discreteMidRanges:          [][]float64{[]float64{1.5, 2.5, 3.5, 4.5}, []float64{1.5, 2.5, 3.5}},
		lower:                      []float64{0.5, 0.5, 0.0, -2.0, 0.0},
		upper:                      []float64{4.5, 3.5, 1.0, 5.0, 1.0},
	}

	// Default no-data logger
	var err error
	cem.data, err = logger.NewData(cem.debug, logger.DataConfig{})
	if err != nil {
		return nil, err
	}

	for _, opt := range opts {
		if err := opt(cem); err != nil {
			return nil, err
		}
	}

	if cem.rng == nil {
		opt := Seed(uint64(time.Now().UnixNano()))
		if err := opt(cem); err != nil {
			return nil, err
		}
	}

	cem.numElite = int64(float64(cem.numSamples) * cem.percentElite)
	cem.numEliteElite = int(cem.numElite / 2.0)

	cem.meanHyperparams = make([]float64, len(cem.hyperparams))
	for i := range cem.meanHyperparams {
		cem.meanHyperparams[i] = (cem.lower[i] + cem.upper[i]) / 2.0
	}

	covariance := mat.NewDense(len(cem.hyperparams), len(cem.hyperparams), nil)
	covarianceRows, covarianceColumns := covariance.Dims()

	minLower := cem.lower[0]
	maxUpper := cem.upper[0]

	for _, v := range cem.lower {
		if v < minLower {
			minLower = v
		}
	}
	for _, v := range cem.upper {
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

	if covariance, err = nearestPD(covariance); err != nil {
		return nil, err
	}

	cem.symmetricCovariance = mat.NewSymDense(len(cem.hyperparams), nil)
	for i := 0; i < len(cem.hyperparams); i++ {
		for j := 0; j < len(cem.hyperparams); j++ {
			cem.symmetricCovariance.SetSym(i, j, (covariance.At(i, j)+covariance.At(j, i))/2.0)
		}
	}

	fmt.Println("Mean :", cem.meanHyperparams)
	fmt.Println("")

	return cem, nil
}

// newSampleSlice creates slices of sampled hyperparams.
// The first returned value contains the original values of hyperparams (discrete, continuous).
// The second returned value contain the continuous representation of hyperparams (continuous).
func (cem Cem) newSampleSlices(cov mat.Cholesky, elitePoints, eliteSamplePoints [][]float64) ([][]float64, [][]float64) {
	samples := make([][]float64, cem.numSamples)
	realvaluedSamples := make([][]float64, cem.numSamples)

	i := 0

	if elitePoints != nil {
		for m := 0; m < int(cem.numEliteElite); m++ {
			realvaluedSamples[m] = elitePoints[m]
			samples[m] = eliteSamplePoints[m]
		}
		i += int(cem.numEliteElite)
	}

	for i < cem.numSamples {
		sample := distmv.NormalRand(nil, cem.meanHyperparams, &cov, rand.NewSource(cem.rng.Uint64()))
		flag := 0
		for j := 0; j < len(cem.hyperparams); j++ {
			if sample[j] < cem.lower[j] || sample[j] > cem.upper[j] {
				flag = 1
				break
			}
		}
		if flag == 0 {
			realvaluedSamples[i] = sample
			var temp []float64
			for j := 0; j < len(cem.hyperparams); j++ {
				if !containsInt(cem.discreteHyperparamsIndices, int64(j)) {
					temp = append(temp, sample[j])
				} else {
					for k := 0; k < len(cem.discreteMidRanges[j]); k++ {
						if sample[j] < cem.discreteMidRanges[indexOfInt(int64(j), cem.discreteHyperparamsIndices)][k] {
							temp = append(temp, cem.discreteRanges[indexOfInt(int64(j), cem.discreteHyperparamsIndices)][k])
							break
						}
					}
				}
			}
			samples[i] = temp
			i++
		}
	}

	return samples, realvaluedSamples
}

func (cem Cem) Run() error {
	var choleskySymmetricCovariance mat.Cholesky
	choleskySymmetricCovariance.Factorize(cem.symmetricCovariance)

	samples, realvaluedSamples := cem.newSampleSlices(choleskySymmetricCovariance, nil, nil)

	// LOG THE MEAN OF THE DISTRIBUTION AFTER EVERY ITERATION

	for iteration := 0; iteration < cem.numIterations; iteration++ {
		startIteration := time.Now()
		fmt.Println("Iteration: ", iteration)
		fmt.Println("")
		samplesMetrics := make([]float64, cem.numSamples)
		fmt.Println("Samples before iteration: ", samples)
		fmt.Println("")

		jobs := make(chan int, cem.numSamples)
		results := make(chan averageAtIndex, cem.numSamples)

		for w := 0; w < cem.numWorkers; w++ {
			go cem.worker(jobs, results, samples, cem.numRuns, iteration)
		}

		for s := 0; s < cem.numSamples; s++ {
			jobs <- s
		}
		close(jobs)

		count := 0
		var err error
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

		elitePoints := make([][]float64, cem.numElite)
		eliteSamplePoints := make([][]float64, cem.numElite)
		elitePoints = descendingRealValuedSamples[:cem.numElite]
		eliteSamplePoints = descendingSamples[:cem.numElite]
		meanSampleHyperparams := make([]float64, len(cem.hyperparams))
		copy(cem.meanHyperparams, elitePoints[0])
		copy(meanSampleHyperparams, eliteSamplePoints[0])
		fmt.Println("Elite points: ", eliteSamplePoints)
		fmt.Println("")
		fmt.Println("Elite Points Metric: ", descendingSamplesMetrics[:cem.numElite])
		fmt.Println("")
		fmt.Println("Mean point: ", meanSampleHyperparams)

		elitePointsMatrix := mat.NewDense(len(elitePoints), len(cem.hyperparams), nil)
		for rows := 0; rows < len(elitePoints); rows++ {
			for cols := 0; cols < len(cem.hyperparams); cols++ {
				elitePointsMatrix.Set(rows, cols, elitePoints[rows][cols])
			}
		}

		cov := mat.NewSymDense(len(cem.hyperparams), nil)
		stat.CovarianceMatrix(cov, elitePointsMatrix, nil)

		covWithE := mat.NewDense(len(cem.hyperparams), len(cem.hyperparams), nil)
		for rows := 0; rows < len(cem.hyperparams); rows++ {
			for cols := 0; cols < len(cem.hyperparams); cols++ {
				if rows == cols {
					covWithE.Set(rows, cols, cov.At(rows, cols)+e)
				} else {
					covWithE.Set(rows, cols, cov.At(rows, cols)-e)
				}
			}
		}

		covariance, err := nearestPD(covWithE)
		if err != nil {
			return err
		}

		cem.symmetricCovariance = mat.NewSymDense(len(cem.hyperparams), nil)
		for i := 0; i < len(cem.hyperparams); i++ {
			for j := 0; j < len(cem.hyperparams); j++ {
				cem.symmetricCovariance.SetSym(i, j, (covariance.At(i, j)+covariance.At(j, i))/2.0)
			}
		}

		var choleskySymmetricCovariance mat.Cholesky
		choleskySymmetricCovariance.Factorize(cem.symmetricCovariance)

		samples, realvaluedSamples = cem.newSampleSlices(choleskySymmetricCovariance, elitePoints, eliteSamplePoints)

		fmt.Println("")
		fmt.Println("Execution time for iteration: ", time.Since(startIteration))
		fmt.Println("")
		fmt.Println("--------------------------------------------------")
	}
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
	var run_metrics []float64
	var run_successes []float64
	for run := 0; run < numRuns; run++ {
		seed := int64((numRuns * iteration) + run)

		set := cem.getSets(seed, sample)

		ag := &agent.ESarsa{Debug: cem.debug}
		ag.InitializeWithSettings(set, lockweight.LockWeight{})

		env := &environment.Acrobot{Debug: cem.debug}
		env.InitializeWithSettings(environment.AcrobotSettings{Seed: seed}) // Episodic acrobot

		expConf := config.Experiment{
			MaxEpisodes:             50000,
			MaxRunLengthEpisodic:    cem.maxRunLengthEpisodic,
			DebugInterval:           0,
			DataPath:                "",
			ShouldLogTraces:         false,
			CacheTracesInRAM:        false,
			ShouldLogEpisodeLengths: false,
			MaxCPUs:                 1,
		}
		exp, err := experiment.New(ag, env, expConf, cem.debug, cem.data)
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
