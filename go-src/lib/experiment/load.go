package experiment

import (
	"errors"

	"github.com/stellentus/cartpoles/go-src/lib/agent"
	"github.com/stellentus/cartpoles/go-src/lib/config"
	"github.com/stellentus/cartpoles/go-src/lib/environment"
	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

// Execute executes the experiment described by the provided JSON.
func Execute(conf config.Config) error {
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
		return errors.New("Could not initialize environment: " + err.Error())
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

func InitializeEnvironment(name string, attr rlglue.Attributes, debug logger.Debug) (rlglue.Environment, error) {
	var err error
	defer debug.Error(&err)

	environment, err := environment.Create(name, debug)
	if err != nil {
		return nil, errors.New("Could not create experiment: " + err.Error())
	}
	err = environment.Initialize(attr)
	if err != nil {
		err = errors.New("Could not initialize experiment: " + err.Error())
	}
	return environment, err
}

func InitializeAgent(name string, attr rlglue.Attributes, env rlglue.Environment, debug logger.Debug) (rlglue.Agent, error) {
	var err error
	defer debug.Error(&err)

	agent, err := agent.Create(name, debug)
	if err != nil {
		return nil, errors.New("Could not create agent: " + err.Error())
	}
	err = agent.Initialize(attr, env.GetAttributes())
	if err != nil {
		err = errors.New("Could not initialize agent: " + err.Error())
	}
	return agent, err
}
