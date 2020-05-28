package experiment

import (
	"errors"
	"strconv"

	"github.com/stellentus/cartpoles/lib/agent"
	"github.com/stellentus/cartpoles/lib/config"
	"github.com/stellentus/cartpoles/lib/environment"
	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

// Execute executes the experiment described by the provided JSON.
func Execute(run uint, conf config.Config) error {
	debugLogger := logger.NewDebug(logger.DebugConfig{
		ShouldPrintDebug: true,
	})
	dataLogger, err := logger.NewData(debugLogger, logger.DataConfig{
		ShouldLogTraces:         conf.Experiment.ShouldLogTraces,
		CacheTracesInRAM:        conf.Experiment.CacheTracesInRAM,
		ShouldLogEpisodeLengths: conf.Experiment.ShouldLogEpisodeLengths,
		BasePath:                conf.Experiment.DataPath,
		FileSuffix:              strconv.Itoa(int(run)),
	})
	if err != nil {
		return errors.New("Could not create data logger: " + err.Error())
	}

	environment, err := InitializeEnvironment(conf.EnvironmentName, run, conf.Environment, debugLogger)
	if err != nil {
		return errors.New("Could not initialize environment: " + err.Error())
	}

	agent, err := InitializeAgent(conf.AgentName, run, conf.Agent, environment, debugLogger)
	if err != nil {
		return err
	}

	expr, err := New(agent, environment, conf.Experiment, debugLogger, dataLogger)
	if err != nil {
		return err
	}

	return expr.Run()
}

func InitializeEnvironment(name string, run uint, attr rlglue.Attributes, debug logger.Debug) (rlglue.Environment, error) {
	var err error
	defer debug.Error(&err)

	environment, err := environment.Create(name, debug)
	if err != nil {
		return nil, errors.New("Could not create experiment: " + err.Error())
	}
	err = environment.Initialize(run, attr)
	if err != nil {
		err = errors.New("Could not initialize experiment: " + err.Error())
	}
	return environment, err
}

func InitializeAgent(name string, run uint, attr rlglue.Attributes, env rlglue.Environment, debug logger.Debug) (rlglue.Agent, error) {
	var err error
	defer debug.Error(&err)

	agent, err := agent.Create(name, debug)
	if err != nil {
		return nil, errors.New("Could not create agent: " + err.Error())
	}
	err = agent.Initialize(run, attr, env.GetAttributes())
	if err != nil {
		err = errors.New("Could not initialize agent: " + err.Error())
	}
	return agent, err
}
