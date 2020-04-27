package registry

import (
	"errors"

	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

// NewAgentCreator is a function that can create an agent.
type NewAgentCreator func(logger.Debug) (rlglue.Agent, error)

// AddAgent is used to register a new agent type.
// This is most likely called by an init function in the Agent's go file.
// The function returns an error if an agent with that name already exists.
func AddAgent(name string, creator NewAgentCreator) error {
	if _, ok := agentList[name]; ok {
		return errors.New("Agent '" + name + "' has already been registered")
	}
	agentList[name] = creator
	return nil
}

func CreateAgent(name string, debug logger.Debug) (rlglue.Agent, error) {
	creator, ok := agentList[name]
	if !ok {
		return nil, errors.New("Agent '" + name + "' has not been registered")
	}
	return creator(debug)
}

var agentList = map[string]NewAgentCreator{}
