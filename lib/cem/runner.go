package cem

import (
	"encoding/json"
	"math"
	"strconv"

	"github.com/stellentus/cartpoles/lib/agent"
	"github.com/stellentus/cartpoles/lib/config"
	"github.com/stellentus/cartpoles/lib/environment"
	"github.com/stellentus/cartpoles/lib/experiment"
	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

type RunnerSettings struct {
	Settings
	Seed               uint64
	ScoreType          RunScorerGenerator
	ExperimentSettings config.Experiment
	EnvironmentSettings
	AgentSettings
}

type Hyper struct {
	Name         string
	Lower, Upper float64
	Discretes    []float64
	IsDiscrete   bool
	IsInt        bool
}

type EnvironmentSettings struct {
	Name     string
	Settings SettingsMap
}

type AgentSettings struct {
	Name       string
	Default    SettingsMap
	CemOptions []Hyper
}

type Runner struct {
	logger.Debug
	logger.Data
	RunnerSettings
}

func NewRunner(set RunnerSettings) (Runner, error) {
	// Set up no-data logger and debug
	debug := logger.NewDebug(logger.DebugConfig{})
	data, err := logger.NewData(debug, logger.DataConfig{})
	if err != nil {
		return Runner{}, err
	}
	return Runner{
		Debug:          debug,
		Data:           data,
		RunnerSettings: set,
	}, nil
}

func (rn Runner) Run(options []Option, datasetSeed uint) ([]float64, error) {
	hypers := []Hyperparameter{}
	for _, hyper := range rn.AgentSettings.CemOptions {
		if hyper.IsDiscrete {
			hypers = append(hypers, NewDiscreteConverter(hyper.Discretes))
		} else {
			hypers = append(hypers, Hyperparameter{Lower: hyper.Lower, Upper: hyper.Upper})
		}
	}

	// If the seed is MaxUint64, then we default to time-based initialization
	if rn.Seed != math.MaxUint64 {
		options = append(options, Seed(rn.Seed))
	}

	cem, err := New(rn.runOneSample, hypers, datasetSeed, rn.Settings, options...)
	if err != nil {
		return nil, err
	}

	return cem.Run(datasetSeed)
}

func (rn Runner) newAgentSettings(hyperparameters []float64) SettingsMap {
	merged := rn.AgentSettings.Default.copy()

	for i, hyp := range rn.CemOptions {
		if hyp.IsInt {
			merged[hyp.Name] = json.RawMessage(strconv.Itoa(int(hyperparameters[i])))
		} else {
			merged[hyp.Name] = json.RawMessage(strconv.FormatFloat(hyperparameters[i], 'E', -1, 64))
		}
	}

	return merged
}

func (rn Runner) attributesWithSeed(set SettingsMap, seed uint64) (rlglue.Attributes, error) {
	set["seed"] = json.RawMessage(strconv.FormatUint(seed, 10))
	attr, err := json.Marshal(set)
	return rlglue.Attributes(attr), err
}

func (rn Runner) runOneSample(hyperparameters []float64, datasetSeed uint, seeds []uint64, iteration int) (float64, error) {
	agSet := rn.newAgentSettings(hyperparameters)
	envSet := rn.EnvironmentSettings.Settings.copy()

	ag, err := agent.Create(rn.AgentSettings.Name, rn.Debug)
	if err != nil {
		return 0, err
	}

	env, err := environment.Create(rn.EnvironmentSettings.Name, rn.Debug)
	if err != nil {
		return 0, err
	}

	scoreGen := rn.ScoreType()

	for run := 0; run < len(seeds); run++ {
		attr, err := rn.attributesWithSeed(agSet, seeds[run])
		if err != nil {
			return 0, err
		}

		ag.Initialize(datasetSeed, attr, nil)

		attr, err = rn.attributesWithSeed(envSet, seeds[run])
		if err != nil {
			return 0, err
		}

		env.Initialize(datasetSeed, attr)

		exp, err := experiment.New(ag, env, rn.ExperimentSettings, rn.Debug, rn.Data)
		if err != nil {
			return 0, err
		}

		listOfListOfRewards, _ := exp.Run()
		scoreGen.UpdateRun(listOfListOfRewards)
	}
	return scoreGen.Score(), nil
}

type SettingsMap map[string]json.RawMessage

func (sm SettingsMap) copy() SettingsMap {
	newSM := make(SettingsMap)
	for key, value := range sm {
		newSM[key] = value
	}
	return newSM
}
