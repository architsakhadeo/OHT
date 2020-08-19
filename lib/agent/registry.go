package agent

import (
	"errors"
	"io"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

// NewPersistentAgentCreator is a function that can create an agent.
type NewPersistentAgentCreator func(logger.Debug) (rlglue.PersistentAgent, error)

var agentList = map[string]NewPersistentAgentCreator{}

// AddPersistent is used to register a new persistent agent type.
func AddPersistent(name string, creator NewPersistentAgentCreator) error {
	if _, ok := agentList[name]; ok {
		return errors.New("Agent '" + name + "' has already been registered")
	}
	agentList[name] = creator
	return nil
}

// Create creates an agent of the provided name.
func Create(name string, debug logger.Debug) (rlglue.PersistentAgent, error) {
	creator, ok := agentList[name]
	if !ok {
		return nil, errors.New("Agent '" + name + "' has not been registered")
	}
	return creator(debug)
}

// NewAgentCreator is a function that can create an agent.
// This is most likely called by an init function in the Agent's go file.
// The function returns an error if an agent with that name already exists.
type NewAgentCreator func(logger.Debug) (rlglue.Agent, error)

// Add is used to register a new agent type.
func Add(name string, creator NewAgentCreator) error {
	persistentCreator := func(debug logger.Debug) (rlglue.PersistentAgent, error) {
		agent, err := creator(debug)
		return persistentAgent{Agent: agent}, err
	}

	return AddPersistent(name, persistentCreator)
}

type persistentAgent struct {
	rlglue.Agent
}

func (pa persistentAgent) Save(wr io.Writer) error {
	return errors.New("Attempt to call Save on rlglue.Agent")
}

func (pa persistentAgent) Load(rd io.Reader) error {
	return errors.New("Attempt to call Load on rlglue.Agent")
}
