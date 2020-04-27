package remote

import (
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

type Agent struct {
	logger rlglue.Logger
}

func NewAgent() rlglue.Agent {
	return &Agent{}
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *Agent) Initialize(expAttr, envAttr rlglue.Attributes, logger rlglue.Logger) error {
	agent.logger = logger
	panic("agent.Initialize not implemented")
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *Agent) Start(state rlglue.State) rlglue.Action {
	panic("agent.Start not implemented")
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *Agent) Step(state rlglue.State, reward float64) rlglue.Action {
	panic("agent.Step not implemented")
}

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *Agent) End(state rlglue.State, reward float64) {
	panic("agent.End not implemented")
}
