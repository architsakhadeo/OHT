package main

import (
	"flag"
	"fmt"
	"math"
	"time"

	"github.com/stellentus/cartpoles/lib/agent"
	"github.com/stellentus/cartpoles/lib/cem"
	"github.com/stellentus/cartpoles/lib/config"
	"github.com/stellentus/cartpoles/lib/environment"
	"github.com/stellentus/cartpoles/lib/experiment"
	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/util/lockweight"
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
	maxRunLengthEpisodic = flag.Int("maxRunLengthEpisodic", 0, "Max number of steps in episode")
	maxEpisodes          = flag.Int("maxEpisodes", 50000, "Max number of episodes")
	percentElite         = flag.Float64("elite", 0.5, "Percent of samples that should be drawn from the elite group")
)

const e = 10.e-8

func main() {
	startTime := time.Now()
	flag.Parse()

	options := []cem.Option{}

	if *seed != math.MaxUint64 {
		options = append(options, cem.Seed(*seed))
	}

	settings := cem.Settings{
		NumWorkers:    *numWorkers,
		NumIterations: *numIterations,
		NumSamples:    *numSamples,
		NumRuns:       *numRuns,
		PercentElite:  *percentElite,
	}

	hypers := []cem.Hyperparameter{
		cem.NewDiscreteConverter([]float64{8, 16, 32, 48}),
		cem.NewDiscreteConverter([]float64{2, 4, 8}),
		cem.Hyperparameter{Lower: 0, Upper: 1},
		cem.Hyperparameter{Lower: -2, Upper: 5},
		cem.Hyperparameter{Lower: 0, Upper: 1},
	}

	rn, err := NewRunner()
	panicIfError(err, "Failed to create Runner")

	cem, err := cem.New(rn.runOneSample, hypers, settings, options...)
	panicIfError(err, "Failed to create CEM")
	err = cem.Run()
	panicIfError(err, "Failed to run CEM")

	fmt.Println("")
	fmt.Println("Execution time: ", time.Since(startTime))
}

type Runner struct {
	logger.Debug
	logger.Data
}

func NewRunner() (Runner, error) {
	// Set up no-data logger and debug
	debug := logger.NewDebug(logger.DebugConfig{})
	data, err := logger.NewData(debug, logger.DataConfig{})
	if err != nil {
		return Runner{}, err
	}
	return Runner{
		Debug: debug,
		Data:  data,
	}, nil
}

func (rn Runner) runOneSample(hyperparameters []float64, seeds []uint64, iteration int) (float64, error) {
	var run_metrics []float64
	var run_successes []float64
	for run := 0; run < len(seeds); run++ {
		set := agent.EsarsaSettings{
			Seed:               int64(seeds[run]),
			NumTilings:         int(hyperparameters[0]),
			NumTiles:           int(hyperparameters[1]),
			Lambda:             float64(hyperparameters[2]),
			WInit:              float64(hyperparameters[3]),
			Alpha:              float64(hyperparameters[4]),
			Gamma:              1.0,
			Epsilon:            0.0,
			AdaptiveAlpha:      0.0,
			IsStepsizeAdaptive: false,
			EnvName:            "acrobot",
		}

		ag := &agent.ESarsa{Debug: rn.Debug}
		ag.InitializeWithSettings(set, lockweight.LockWeight{})

		env := &environment.Acrobot{Debug: rn.Debug}
		env.InitializeWithSettings(environment.AcrobotSettings{Seed: int64(seeds[run])}) // Episodic acrobot

		expConf := config.Experiment{
			MaxEpisodes:          *maxEpisodes,
			MaxRunLengthEpisodic: *maxRunLengthEpisodic,
		}
		exp, err := experiment.New(ag, env, expConf, rn.Debug, rn.Data)
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

func panicIfError(err error, reason string) {
	if err != nil {
		panic("ERROR " + err.Error() + ": " + reason)
	}
}
