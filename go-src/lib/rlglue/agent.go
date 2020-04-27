package rlglue

import "errors"

type Agent interface {
	// Initialize configures the agent with the provided parameters and resets any internal state.
	// The first attributes are experimental attributes; the second are environmental.
	Initialize(Attributes, Attributes, Logger) error

	// Start provides an initial observation to the agent and returns the agent's action.
	Start(state State) Action

	// Step provides a new observation and a reward to the agent and returns the agent's next action.
	Step(state State, reward float64) Action

	// End informs the agent that a terminal state has been reached, providing the final reward.
	End(state State, reward float64)
}

// NewAgentCreator is a function that can create an agent.
type NewAgentCreator func() (Agent, error)

// RegisterAgent is used to register a new agent type.
// This is most likely called by an init function in the Agent's go file.
// The function returns an error if an agent with that name already exists.
func RegisterAgent(name string, creator NewAgentCreator) error {
	if _, ok := agentList[name]; ok {
		return errors.New("Agent '" + name + "' has already been registered")
	}
	agentList[name] = creator
	return nil
}

func CreateAgent(name string) (Agent, error) {
	creator, ok := agentList[name]
	if !ok {
		return nil, errors.New("Agent '" + name + "' has not been registered")
	}
	return creator()
}

var agentList = map[string]NewAgentCreator{}
