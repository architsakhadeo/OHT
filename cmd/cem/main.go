package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/stellentus/cartpoles/lib/agent"
	"github.com/stellentus/cartpoles/lib/cem"
	"github.com/stellentus/cartpoles/lib/config"
	"github.com/stellentus/cartpoles/lib/environment"
	"github.com/stellentus/cartpoles/lib/experiment"
	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

var (
	configPath = flag.String("cem", "config/cem/cem.json", "CEM settings file path")
	expPath    = flag.String("exp", "config/cem/experiment.json", "Experiment settings file path")
	agentPath  = flag.String("agent", "config/cem/agent.json", "Default agent settings file path")
)

type agentSettings struct {
	Name       string
	Default    map[string]json.RawMessage
	CemOptions []hyper
}

type cemSettings struct {
	cem.Settings
	Seed       uint64
	experiment config.Experiment
	agentSettings
}

type hyper struct {
	Name         string
	Lower, Upper float64
	Discretes    []float64
	IsDiscrete   bool
	IsInt        bool
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

	rn, err := NewRunner(settings.experiment, settings.agentSettings)
	panicIfError(err, "Failed to create Runner")

	hypers := []cem.Hyperparameter{}
	for _, hyper := range settings.agentSettings.CemOptions {
		if hyper.IsDiscrete {
			hypers = append(hypers, cem.NewDiscreteConverter(hyper.Discretes))
		} else {
			hypers = append(hypers, cem.Hyperparameter{Lower: hyper.Lower, Upper: hyper.Upper})
		}
	}

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
	agentSettings
}

func NewRunner(exp config.Experiment, as agentSettings) (Runner, error) {
	// Set up no-data logger and debug
	debug := logger.NewDebug(logger.DebugConfig{})
	data, err := logger.NewData(debug, logger.DataConfig{})
	if err != nil {
		return Runner{}, err
	}
	return Runner{
		Debug:         debug,
		Data:          data,
		Experiment:    exp,
		agentSettings: as,
	}, nil
}

func (rn Runner) newSettings(hyperparameters []float64) map[string]json.RawMessage {
	merged := make(map[string]json.RawMessage)

	// Copy from the original map to the target map
	for key, value := range rn.agentSettings.Default {
		merged[key] = value
	}

	for i, hyp := range rn.CemOptions {
		if hyp.IsInt {
			merged[hyp.Name] = json.RawMessage(strconv.Itoa(int(hyperparameters[i])))
		} else {
			merged[hyp.Name] = json.RawMessage(strconv.FormatFloat(hyperparameters[i], 'E', -1, 64))
		}
	}

	return merged
}

func (rn Runner) attributesWithSeed(set map[string]json.RawMessage, seed uint64) (rlglue.Attributes, error) {
	set["seed"] = json.RawMessage(strconv.FormatUint(seed, 10))
	attr, err := json.Marshal(set)
	return rlglue.Attributes(attr), err
}

func (rn Runner) runOneSample(hyperparameters []float64, seeds []uint64, iteration int) (float64, error) {
	average := 0
	average_success := 0
	set := rn.newSettings(hyperparameters)

	ag, err := agent.Create(rn.agentSettings.Name, rn.Debug)
	if err != nil {
		return 0, err
	}

	for run := 0; run < len(seeds); run++ {
		attr, err := rn.attributesWithSeed(set, seeds[run])
		if err != nil {
			return 0, err
		}

		ag.Initialize(0, attr, nil)

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
	readJsonFile(*agentPath, &settings.agentSettings)

	return settings
}

func readJsonFile(path string, val interface{}) {
	data, err := ioutil.ReadFile(path)
	panicIfError(err, "Couldn't load config file '"+path+"'")
	err = json.Unmarshal(data, val)
	panicIfError(err, "Couldn't parse config JSON '"+string(data)+"'")
}
