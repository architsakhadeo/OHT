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

	numHyperparams             int
	discreteHyperparamsIndices []int64
	discreteRanges             [][]float64
	discreteMidRanges          [][]float64
	numElite                   int
	meanHyperparams            []float64
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
		numHyperparams:             5,
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
		if err := opt.apply(cem); err != nil {
			return nil, err
		}
	}

	if cem.rng == nil {
		opt := Seed(uint64(time.Now().UnixNano()))
		if err := opt.apply(cem); err != nil {
			return nil, err
		}
	}

	cem.numElite = int(float64(cem.numSamples) * cem.percentElite)

	cem.meanHyperparams = make([]float64, cem.numHyperparams)
	for i := range cem.meanHyperparams {
		cem.meanHyperparams[i] = (cem.lower[i] + cem.upper[i]) / 2.0
	}

	fmt.Println("Mean :", cem.meanHyperparams)
	fmt.Println("")

	return cem, nil
}

func (cem Cem) initialCovariance() *mat.Dense {
	covariance := mat.NewDense(cem.numHyperparams, cem.numHyperparams, nil)
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

	return covariance
}

// setSamples returns a valid set of samples.
// It loops until the hyperparameters meet the required lower/upper constraint.
func (cem Cem) setSamples(chol *mat.Cholesky, samples *mat.Dense, row int) []float64 {
	for true {
		ok := true
		sample := distmv.NormalRand(nil, cem.meanHyperparams, chol, rand.NewSource(cem.rng.Uint64()))
		for j := 0; j < cem.numHyperparams; j++ {
			if sample[j] < cem.lower[j] || sample[j] > cem.upper[j] {
				ok = false
				break
			}
		}
		if ok {
			samples.SetRow(row, sample)
			return sample
		}
	}
	return nil // code cannot reach this point
}

// newSampleSlice creates slices of sampled hyperparams.
// The first returned value contains the original values of hyperparams (discrete, continuous).
// The second returned value contain the continuous representation of hyperparams (continuous).
func (cem Cem) newSampleSlices(covariance *mat.Dense, elitesRealVals, elites *mat.Dense) (*mat.Dense, *mat.Dense, error) {
	chol, err := choleskySymmetricFromCovariance(covariance, cem.numHyperparams)
	if err != nil {
		return nil, nil, err
	}

	samples := mat.NewDense(cem.numSamples, cem.numHyperparams, nil)
	samplesRealVals := mat.NewDense(cem.numSamples, cem.numHyperparams, nil)

	i := 0

	if elitesRealVals != nil {
		numEliteElite := cem.numElite / 2
		for m := 0; m < numEliteElite; m++ {
			samplesRealVals.SetRow(m, elitesRealVals.RawRowView(m))
			samples.SetRow(m, elites.RawRowView(m))
		}
		i += numEliteElite
	}

	for ; i < cem.numSamples; i++ {
		cem.setSamples(chol, samplesRealVals, i)

		for j := 0; j < cem.numHyperparams; j++ {
			if !containsInt(cem.discreteHyperparamsIndices, int64(j)) {
				samples.Set(i, j, samplesRealVals.At(i, j))
			} else {
				for k := 0; k < len(cem.discreteMidRanges[j]); k++ {
					if samplesRealVals.At(i, j) < cem.discreteMidRanges[indexOfInt(int64(j), cem.discreteHyperparamsIndices)][k] {
						samples.Set(i, j, cem.discreteRanges[indexOfInt(int64(j), cem.discreteHyperparamsIndices)][k])
						break
					}
				}
			}
		}
	}

	return samples, samplesRealVals, nil
}

func (cem Cem) Run() error {
	covariance := cem.initialCovariance()

	samples, samplesRealVals, err := cem.newSampleSlices(covariance, nil, nil)
	if err != nil {
		return err
	}

	// Allocate memory outside of loop
	descendingSamplesMetrics := make([]float64, cem.numElite)
	elitesRealVals := mat.NewDense(cem.numElite, cem.numHyperparams, nil)
	elites := mat.NewDense(cem.numElite, cem.numHyperparams, nil)
	meanSampleHyperparams := make([]float64, cem.numHyperparams)
	samplesMetrics := make([]float64, cem.numSamples)

	for iteration := 0; iteration < cem.numIterations; iteration++ {
		startIteration := time.Now()
		fmt.Println("Iteration: ", iteration)
		fmt.Println("")
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
		for count < cem.numSamples {
			select {
			case avg := <-results:
				if avg.err != nil {
					err = avg.err // Only the most recent will be returned
				}
				count++
				samplesMetrics[avg.idx] = avg.average
			}
		}
		close(results)
		if err != nil {
			return err
		}

		fmt.Println("Sample Metric: ", samplesMetrics)
		fmt.Println("")
		ascendingIndices := argsort.Sort(sort.Float64Slice(samplesMetrics))
		for ind := 0; ind < cem.numElite; ind++ {
			descendingSamplesMetrics[ind] = samplesMetrics[ascendingIndices[cem.numSamples-1-ind]]
			elites.SetRow(ind, samples.RawRowView(ascendingIndices[cem.numSamples-1-ind]))
			elitesRealVals.SetRow(ind, samplesRealVals.RawRowView(ascendingIndices[cem.numSamples-1-ind]))
		}

		copy(cem.meanHyperparams, elitesRealVals.RawRowView(0))
		copy(meanSampleHyperparams, elites.RawRowView(0))
		fmt.Println("Elite points: ", elites)
		fmt.Println("")
		fmt.Println("Elite Points Metric: ", descendingSamplesMetrics[:cem.numElite])
		fmt.Println("")
		fmt.Println("Mean point: ", meanSampleHyperparams)

		elitesRealValsMatrix := mat.NewDense(cem.numElite, cem.numHyperparams, nil)
		for rows := 0; rows < cem.numElite; rows++ {
			for cols := 0; cols < cem.numHyperparams; cols++ {
				elitesRealValsMatrix.Set(rows, cols, elitesRealVals.At(rows, cols))
			}
		}

		cov := mat.NewSymDense(cem.numHyperparams, nil)
		stat.CovarianceMatrix(cov, elitesRealValsMatrix, nil)

		covariance = mat.NewDense(cem.numHyperparams, cem.numHyperparams, nil)
		for rows := 0; rows < cem.numHyperparams; rows++ {
			for cols := 0; cols < cem.numHyperparams; cols++ {
				if rows == cols {
					covariance.Set(rows, cols, cov.At(rows, cols)+e)
				} else {
					covariance.Set(rows, cols, cov.At(rows, cols)-e)
				}
			}
		}

		samples, samplesRealVals, err = cem.newSampleSlices(covariance, elitesRealVals, elites)
		if err != nil {
			return err
		}

		fmt.Println("")
		fmt.Println("Execution time for iteration: ", time.Since(startIteration))
		fmt.Println("")
		fmt.Println("--------------------------------------------------")
	}
	return nil
}

func (cem Cem) worker(jobs <-chan int, results chan<- averageAtIndex, samples *mat.Dense, numRuns, iteration int) {
	for idx := range jobs {
		average, err := cem.runOneSample(samples.RawRowView(idx), numRuns, iteration)
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
			MaxEpisodes:          50000,
			MaxRunLengthEpisodic: cem.maxRunLengthEpisodic,
		}
		exp, err := experiment.New(ag, env, expConf, cem.debug, cem.data)
		if err != nil {
			return 0, err
		}

		listOfListOfRewards, _ := exp.Run()
		var listOfRewards []float64

		//Episodic Acrobot, last 1/10th of the episodes
		for i := 0; i < len(listOfListOfRewards); i++ {
			listOfRewards = append(listOfRewards, listOfListOfRewards[i]...)
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
