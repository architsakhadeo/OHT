package experiment

import (
	"encoding/json"
	"errors"

	"github.com/stellentus/cartpoles/go-src/lib/agent"
	"github.com/stellentus/cartpoles/go-src/lib/environment"
	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

// Execute executes the experiment described by the provided JSON.
func Execute(data json.RawMessage) error {
	conf := Config{
		Experiment: Settings{
			MaxEpisodes:             0,
			MaxSteps:                0,
			DebugInterval:           1,
			DataPath:                "",
			ShouldLogTraces:         false,
			ShouldLogEpisodeLengths: false,
		},
	}
	err := json.Unmarshal(data, &conf)
	if err != nil {
		return errors.New("The config file is not valid JSON: " + err.Error())
	}

	debugLogger := logger.NewDebug(logger.DebugConfig{
		ShouldPrintDebug: true,
	})
	dataLogger, err := logger.NewData(debugLogger, logger.DataConfig{
		ShouldLogTraces:         conf.Experiment.ShouldLogTraces,
		ShouldLogEpisodeLengths: conf.Experiment.ShouldLogEpisodeLengths,
		BasePath:                conf.Experiment.DataPath,
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

	expr, err := New(agent, environment, conf.Experiment, debugLogger, dataLogger)
	if err != nil {
		return err
	}

	return expr.Run()
}

type Settings struct {
	MaxEpisodes             int    `json:"episodes"`
	MaxSteps                int    `json:"steps"`
	DebugInterval           int    `json:"debug-interval"`
	DataPath                string `json:"data-path"`
	ShouldLogTraces         bool   `json:"should-log-traces"`
	ShouldLogEpisodeLengths bool   `json:"should-log-episode-lengths"`
}

type Config struct {
	EnvironmentName string            `json:"environment-name"`
	AgentName       string            `json:"agent-name"`
	Environment     rlglue.Attributes `json:"environment-settings"`
	Agent           rlglue.Attributes `json:"agent-settings"`
	Experiment      Settings          `json:"experiment-settings"`
}

func InitializeEnvironment(name string, attr rlglue.Attributes, debug logger.Debug) (rlglue.Environment, error) {
	var err error
	defer debug.Error(&err)

	environment, err := environment.Create(name, debug)
	if err != nil {
		return nil, err
	}
	err = environment.Initialize(attr)
	return environment, err
}

func InitializeAgent(name string, attr rlglue.Attributes, env rlglue.Environment, debug logger.Debug) (rlglue.Agent, error) {
	var err error
	defer debug.Error(&err)

	agent, err := agent.Create(name, debug)
	if err != nil {
		return nil, err
	}
	err = agent.Initialize(attr, env.GetAttributes())
	return agent, err
}
