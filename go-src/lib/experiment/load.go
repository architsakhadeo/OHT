package experiment

import (
	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/registry"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

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
