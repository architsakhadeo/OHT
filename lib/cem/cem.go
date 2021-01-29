package cem

import (
	"errors"
	"fmt"
	"io"
	"math"
	"runtime"
	"sort"
	"time"

	"github.com/mkmik/argsort"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distmv"
)

type Settings struct {
	// NumWorkers is the maximum number of workers
	// Default is the number of CPUs if -1
	NumWorkers int

	// NumIterations is the total number of iterations
	NumIterations int

	// NumSamples is the number of samples per iteration
	NumSamples int

	// NumRuns is the number of runs per sample
	NumRuns int

	// PercentElite is the percent of samples that should be drawn from the elite group
	PercentElite float64
}

type Cem struct {
	run RunFunc
	rng *rand.Rand

	numHyperparams int
	numElite       int

	hypers []Hyperparameter

	Settings

	debugWriter io.Writer
}

// SampleValueConverter converts a sample to a value.
type SampleValueConverter interface {
	RealValue(val float64) float64
}

type Hyperparameter struct {
	Lower                float64
	Upper                float64
	SampleValueConverter // optional
}

// RunFunc is a function that runs the code to be optimized, returning its score
type RunFunc func(hyperparameters []float64, seeds []uint64, iteration int) (float64, error)

const e = 10.e-8

func DefaultSettings() Settings {
	return Settings{
		NumWorkers:    runtime.NumCPU(),
		NumIterations: 3,
		NumSamples:    10,
		NumRuns:       2,
		PercentElite:  0.5,
	}
}

func New(run RunFunc, hypers []Hyperparameter, settings Settings, opts ...Option) (*Cem, error) {
	if run == nil {
		return nil, errors.New("Cem requires a run function")
	}
	if len(hypers) == 0 {
		return nil, errors.New("Cem requires hyperparameters")
	}
	if settings.NumWorkers <= 0 {
		settings.NumWorkers = runtime.NumCPU()
	}

	cem := &Cem{
		run:            run,
		hypers:         hypers,
		numHyperparams: len(hypers),
		Settings:       settings,
		numElite:       int(float64(settings.NumSamples) * settings.PercentElite),
		debugWriter:    noopWriter,
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

	return cem, nil
}

func (cem Cem) initialCovariance() *mat.Dense {
	covariance := mat.NewDense(cem.numHyperparams, cem.numHyperparams, nil)
	covarianceRows, covarianceColumns := covariance.Dims()

	minLower := cem.hypers[0].Lower
	maxUpper := cem.hypers[0].Upper

	for _, hy := range cem.hypers {
		if hy.Lower < minLower {
			minLower = hy.Lower
		}
		if hy.Upper > maxUpper {
			maxUpper = hy.Upper
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
func (cem Cem) setSamples(chol *mat.Cholesky, samples *mat.Dense, row int, means []float64) []float64 {
	for true {
		ok := true
		sample := distmv.NormalRand(nil, means, chol, rand.NewSource(cem.rng.Uint64()))
		for j := 0; j < cem.numHyperparams; j++ {
			if sample[j] < cem.hypers[j].Lower || sample[j] >= cem.hypers[j].Upper {
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
func (cem Cem) newSampleSlices(covariance, samples, samplesRealVals *mat.Dense, startIdx int, means []float64) error {
	chol, err := choleskySymmetricFromCovariance(covariance, cem.numHyperparams)
	if err != nil {
		return err
	}

	for i := startIdx; i < cem.NumSamples; i++ {
		cem.setSamples(chol, samplesRealVals, i, means)
		samples.SetRow(i, samplesRealVals.RawRowView(i))
	}

	// Ensure the discrete samples are handled, if necessary
	cem.updateDiscretes(startIdx, samples, samplesRealVals)

	return nil
}

func (cem Cem) updateDiscretes(startRow int, samples, samplesRealVals *mat.Dense) {
	for col := 0; col < cem.numHyperparams; col++ {
		conv := cem.hypers[col].SampleValueConverter
		if conv == nil {
			continue // no need to handle this column
		}
		for row := startRow; row < cem.NumSamples; row++ {
			samples.Set(row, col, conv.RealValue(samplesRealVals.At(row, col)))
		}
	}
}

func (cem Cem) Run() error {
	// Allocate memory outside of loop
	samples := mat.NewDense(cem.NumSamples, cem.numHyperparams, nil)
	samplesRealVals := mat.NewDense(cem.NumSamples, cem.numHyperparams, nil)
	descendingSamplesMetrics := make([]float64, cem.numElite)
	elitesRealVals := mat.NewDense(cem.numElite, cem.numHyperparams, nil)
	elites := mat.NewDense(cem.numElite, cem.numHyperparams, nil)
	samplesMetrics := make([]float64, cem.NumSamples)

	covariance := cem.initialCovariance()

	// Store mean values in elitesRealVals.RawRowView(0), since the starting distribution is centered around it
	for col := 0; col < cem.numHyperparams; col++ {
		elitesRealVals.Set(0, col, (cem.hypers[col].Lower+cem.hypers[col].Upper)/2.0)
	}

	if cem.debugWriter != noopWriter {
		fmt.Fprintf(cem.debugWriter, "Mean: %v\n\n", elitesRealVals.RawRowView(0))
	}

	numEliteElite := 0 // At first there are no elite samples

	for iteration := 0; iteration < cem.NumIterations; iteration++ {
		startIteration := time.Now()
		if cem.debugWriter != noopWriter {
			fmt.Fprintf(cem.debugWriter, "Iteration: %d\n\n", iteration)
		}

		err := cem.newSampleSlices(covariance, samples, samplesRealVals, numEliteElite, elitesRealVals.RawRowView(0))
		if err != nil {
			return err
		}

		if cem.debugWriter != noopWriter {
			fmt.Fprintf(cem.debugWriter, "Samples before iteration: %v\n\n", samples)
		}

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
		for count < cem.NumSamples {
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

		// TODO all of this could be done with far less copying. Each row's backing data could be put into the new matrix.

		if cem.debugWriter != noopWriter {
			fmt.Fprintf(cem.debugWriter, "Sample Metric: %v\n\n", samplesMetrics)
		}
		ascendingIndices := argsort.Sort(sort.Float64Slice(samplesMetrics))
		for ind := 0; ind < cem.numElite; ind++ {
			descendingSamplesMetrics[ind] = samplesMetrics[ascendingIndices[cem.NumSamples-1-ind]]
			elites.SetRow(ind, samples.RawRowView(ascendingIndices[cem.NumSamples-1-ind]))
			elitesRealVals.SetRow(ind, samplesRealVals.RawRowView(ascendingIndices[cem.NumSamples-1-ind]))
		}

		numEliteElite = cem.numElite / 2 // For every iteration after the first, we have some elite
		for m := 0; m < numEliteElite; m++ {
			samplesRealVals.SetRow(m, elitesRealVals.RawRowView(m))
			samples.SetRow(m, elites.RawRowView(m))
		}

		if cem.debugWriter != noopWriter {
			fmt.Fprintf(cem.debugWriter, "Elite points: %v\n\n", elites)
			fmt.Fprintf(cem.debugWriter, "Elite Points Metric: %v\n\n", descendingSamplesMetrics[:cem.numElite])
			fmt.Fprintf(cem.debugWriter, "Mean point: %v\n\n", elites.RawRowView(0))
		}

		cov := mat.NewSymDense(cem.numHyperparams, nil)
		stat.CovarianceMatrix(cov, elitesRealVals, nil)

		covariance = mat.NewDense(cem.numHyperparams, cem.numHyperparams, nil)
		for row := 0; row < cem.numHyperparams; row++ {
			for col := 0; col < cem.numHyperparams; col++ {
				if row == col {
					covariance.Set(row, col, cov.At(row, col)+e)
				} else {
					covariance.Set(row, col, cov.At(row, col)-e)
				}
			}
		}

		if cem.debugWriter != noopWriter {
			fmt.Fprintf(cem.debugWriter, "Execution time for iteration: %v\n\n", time.Since(startIteration))
			fmt.Fprintf(cem.debugWriter, "--------------------------------------------------\n")
		}
	}

	return nil
}

func (cem Cem) worker(jobs <-chan int, results chan<- averageAtIndex, samples *mat.Dense, NumRuns, iteration int) {
	seeds := make([]uint64, NumRuns)
	for i := range seeds {
		seeds[i] = uint64((NumRuns * iteration) + i)
	}

	for idx := range jobs {
		average, err := cem.run(samples.RawRowView(idx), seeds, iteration)
		results <- averageAtIndex{
			average: average,
			idx:     idx,
			err:     err,
		}
	}
}

func containsInt(s []int, e int) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}

// Works if data has unique elements without repetition
func indexOfInt(element int, data []int) int {
	for k, v := range data {
		if element == v {
			return k
		}
	}
	return -1 //not found.
}

type averageAtIndex struct {
	average float64
	idx     int
	err     error
}

func NewDiscreteConverter(vals []float64) Hyperparameter {
	return Hyperparameter{
		Lower:                0.5,
		Upper:                float64(len(vals)) + .5,
		SampleValueConverter: discreteConverter(vals),
	}
}

type discreteConverter []float64

func (dc discreteConverter) RealValue(val float64) float64 {
	for k := 0; k < len(dc); k++ {
		if val <= float64(k)+1.5 {
			return dc[k]
		}
	}
	return dc[len(dc)-1] // Never happens, but just in case
}

var noopWriter = nw{}

type nw struct{}

func (nw nw) Write(p []byte) (n int, err error) { return 0, nil }
