package experiment

import (
	"encoding/json"
	"errors"

	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/registry"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

func Execute(data []byte) error {
	var conf Config
	err := json.Unmarshal(data, &conf)
	if err != nil {
		return errors.New("The config file is not valid JSON: " + err.Error())
	}

	debugLogger := logger.NewDebug(logger.DebugConfig{
		ShouldPrintDebug: true,
		Interval:         2,
	})
	dataLogger := logger.NewData(debugLogger, logger.DataConfig{
		ShouldLogTraces:         false,
		ShouldLogEpisodeLengths: true,
		NumberOfSteps:           1000,                     // TODO how to load this?
		BasePath:                "/save/here/from/config", // TODO
		FileSuffix:              "",                       // TODO after figuring out runs
	})

	expr, err := New(conf.Experiment, conf.Agent, conf.Environment, debugLogger, dataLogger)
	if err != nil {
		return err
	}

	expr.Run()
	return nil
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
