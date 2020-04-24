package example

import (
	"math/rand"

	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

// Agent just iterates through all actions, starting from a random one.
type Agent struct {
	logger     rlglue.Logger
	lastAction int
	attr       rlglue.EnvironmentAttributes
}

func NewAgent() rlglue.Agent {
	return &Agent{}
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *Agent) Initialize(config rlglue.Config, attr rlglue.EnvironmentAttributes, logger rlglue.Logger) {
	agent.logger = logger
	agent.attr = attr

	var seed int64
	if sd, ok := config["seed"]; !ok {
		// Config doesn't have a seed
		seed = 0
	} else if seed, ok = sd.(int64); !ok {
		// Config seed is wrong type
		logger.Message("example.Agent seed was of wrong type")
		seed = 0
	}
	rand.Seed(seed)
	agent.lastAction = rand.Intn(attr.NumberOfActions)
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *Agent) Start(state rlglue.State) rlglue.Action {
	return agent.Step(0, state)
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *Agent) Step(reward float64, state rlglue.State) rlglue.Action {
	agent.lastAction++
	if agent.lastAction > agent.attr.NumberOfActions {
		agent.lastAction = 0
	}
	return rlglue.Action(agent.lastAction)
}

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *Agent) End(reward float64) {}
