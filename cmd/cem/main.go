package main

import (
	"flag"
	"fmt"
	"math"
	"time"

	"github.com/stellentus/cartpoles/lib/cem"
)

var (
	seed                 = flag.Uint64("seed", math.MaxUint64, "Seed to use; if 0xffffffffffffffff, use the time")
	numWorkers           = flag.Int("workers", -1, "Maximum number of workers; defaults to the number of CPUs if -1")
	numIterations        = flag.Int("iterations", 3, "Total number of iterations")
	numSamples           = flag.Int("samples", 10, "Number of samples per iteration")
	numRuns              = flag.Int("runs", 2, "Number of runs per sample")
	numTimesteps         = flag.Int("timesteps", 0, "Number of timesteps per run")
	numEpisodes          = flag.Int("episodes", -1, "Number of episodes")
	numStepsInEpisode    = flag.Int("stepsInEpisode", -1, "Number of steps in episode")
	MaxRunLengthEpisodic = flag.Int("maxRunLengthEpisodic", 0, "Max number of steps in episode")
	percentElite         = flag.Float64("elite", 0.5, "Percent of samples that should be drawn from the elite group")
)

const e = 10.e-8

func main() {
	startTime := time.Now()
	flag.Parse()

	options := []cem.Option{
		cem.NumIterations(*numIterations),
		cem.NumSamples(*numSamples),
		cem.NumRuns(*numRuns),
		cem.NumTimesteps(*numTimesteps),
		cem.NumEpisodes(*numEpisodes),
		cem.NumStepsInEpisode(*numStepsInEpisode),
		cem.MaxRunLengthEpisodic(*MaxRunLengthEpisodic),
		cem.PercentElite(*percentElite),
	}

	if *seed != math.MaxUint64 {
		options = append(options, cem.Seed(*seed))
	}
	if *numWorkers != -1 {
		options = append(options, cem.NumWorkers(*numWorkers))
	}

	cem, err := cem.New(options...)
	panicIfError(err, "Failed to create CEM")
	err = cem.Run()
	panicIfError(err, "Failed to run CEM")

	fmt.Println("")
	fmt.Println("Execution time: ", time.Since(startTime))
}

func panicIfError(err error, reason string) {
	if err != nil {
		panic("ERROR " + err.Error() + ": " + reason)
	}
}
