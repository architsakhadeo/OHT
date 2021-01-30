package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"os"
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
	configPath = flag.String("cem", "config/cem/cem.json", "CEM settings file path")
	expPath    = flag.String("exp", "config/cem/experiment.json", "Experiment settings file path")
)

type cemSettings struct {
	cem.Settings
	Seed       uint64
	experiment config.Experiment
}

const e = 10.e-8

func main() {
	startTime := time.Now()
	flag.Parse()

	settings := buildSettings()

	options := []cem.Option{cem.Debug(os.Stdout)}

	// If the seed is MaxUint64, then we default to time-based initialization
	if settings.Seed != math.MaxUint64 {
		options = append(options, cem.Seed(settings.Seed))
	}

	hypers := []cem.Hyperparameter{
		cem.NewDiscreteConverter([]float64{8, 16, 32, 48}),
		cem.NewDiscreteConverter([]float64{2, 4, 8}),
		cem.Hyperparameter{Lower: 0, Upper: 1},
		cem.Hyperparameter{Lower: -2, Upper: 5},
		cem.Hyperparameter{Lower: 0, Upper: 1},
	}

	rn, err := NewRunner(settings.experiment)
	panicIfError(err, "Failed to create Runner")

	cem, err := cem.New(rn.runOneSample, hypers, settings.Settings, options...)
	panicIfError(err, "Failed to create CEM")

	result, err := cem.Run()
	panicIfError(err, "Failed to run CEM")

	fmt.Println("\nFinal optional point: ", result)
	fmt.Println("Execution time: ", time.Since(startTime))
}

type Runner struct {
	logger.Debug
	logger.Data
	config.Experiment
}

func NewRunner(exp config.Experiment) (Runner, error) {
	// Set up no-data logger and debug
	debug := logger.NewDebug(logger.DebugConfig{})
	data, err := logger.NewData(debug, logger.DataConfig{})
	if err != nil {
		return Runner{}, err
	}
	return Runner{
		Debug:      debug,
		Data:       data,
		Experiment: exp,
	}, nil
}

func (rn Runner) runOneSample(hyperparameters []float64, seeds []uint64, iteration int) (float64, error) {
	average := 0
	average_success := 0
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

		exp, err := experiment.New(ag, env, rn.Experiment, rn.Debug, rn.Data)
		if err != nil {
			return 0, err
		}

		listOfListOfRewards, _ := exp.Run()
		average_success += len(listOfListOfRewards)

		//Episodic Acrobot, last 1/10th of the episodes
		for i := 0; i < len(listOfListOfRewards); i++ {
			average += len(listOfListOfRewards[i])
		}
	}
	average_steps_to_failure := float64(average) / float64(average_success)
	return -average_steps_to_failure, nil //episodic  acrobot, returns negative of steps to failure
}

func panicIfError(err error, reason string) {
	if err != nil {
		panic("ERROR " + err.Error() + ": " + reason)
	}
}

func buildSettings() cemSettings {
	// Build default settings
	settings := cemSettings{
		Seed: math.MaxUint64,
	}
	settings.Settings = cem.DefaultSettings()

	readJsonFile(*configPath, &settings)
	readJsonFile(*expPath, &settings.experiment)

	return settings
}

func readJsonFile(path string, val interface{}) {
	data, err := ioutil.ReadFile(path)
	panicIfError(err, "Couldn't load config file '"+path+"'")
	err = json.Unmarshal(data, val)
	panicIfError(err, "Couldn't parse config JSON '"+string(data)+"'")
}
