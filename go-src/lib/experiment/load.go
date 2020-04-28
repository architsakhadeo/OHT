package experiment

import (
	"encoding/json"
	"errors"

	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/registry"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

// Execute executes the experiment described by the provided JSON.
func Execute(data json.RawMessage) error {
	var conf Config
	err := json.Unmarshal(data, &conf)
	if err != nil {
		return errors.New("The config file is not valid JSON: " + err.Error())
	}

	// Parse settings
	set := settings{
		MaxEpisodes:   0,
		MaxSteps:      0,
		DebugInterval: 1,
		DataPath:      "",
	}
	err = json.Unmarshal(conf.Experiment, &set)
	if err != nil {
		err = errors.New("Experiment settings couldn't be parsed: " + err.Error())
		return err
	}

	debugLogger := logger.NewDebug(logger.DebugConfig{
		ShouldPrintDebug: true,
	})
	dataLogger, err := logger.NewData(debugLogger, logger.DataConfig{
		ShouldLogTraces:         false,
		ShouldLogEpisodeLengths: true,
		NumberOfSteps:           1000, // TODO how to load this?
		BasePath:                set.DataPath,
		FileSuffix:              "", // TODO after figuring out runs
	})
	if err != nil {
		return errors.New("Could not create data logger: " + err.Error())
	}

	environment, err := InitializeEnvironment(conf.EnvironmentName, conf.Environment, debugLogger)
	if err != nil {
		return err
	}

	agent, err := InitializeAgent(conf.AgentName, conf.Agent, environment, debugLogger)
	if err != nil {
		return err
	}

	expr, err := New(agent, environment, set, debugLogger, dataLogger)
	if err != nil {
		return err
	}

	expr.Run()
	return nil
}

type settings struct {
	MaxEpisodes   int    `json:"episodes"`
	MaxSteps      int    `json:"steps"`
	DebugInterval int    `json:"debug-interval"`
	DataPath      string `json:"data-path"`
}

type Config struct {
	EnvironmentName string            `json:"environment-name"`
	AgentName       string            `json:"agent-name"`
	Environment     rlglue.Attributes `json:"environment-settings"`
	Agent           rlglue.Attributes `json:"agent-settings"`
	Experiment      json.RawMessage   `json:"experiment-settings"`
}

func InitializeEnvironment(name string, attr rlglue.Attributes, debug logger.Debug) (rlglue.Environment, error) {
	var err error
	defer debug.Error(&err)

	environment, err := registry.CreateEnvironment(name, debug)
	if err != nil {
		return nil, err
	}
	err = environment.Initialize(attr)
	return environment, err
}

func InitializeAgent(name string, attr rlglue.Attributes, env rlglue.Environment, debug logger.Debug) (rlglue.Agent, error) {
	var err error
	defer debug.Error(&err)

	agent, err := registry.CreateAgent(name, debug)
	if err != nil {
		return nil, err
	}
	err = agent.Initialize(attr, env.GetAttributes())
	return agent, err
}
