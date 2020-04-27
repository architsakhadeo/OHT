package example

import (
	"encoding/json"
	"math/rand"

	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/registry"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

// Agent just iterates through all actions, starting from a random one.
type Agent struct {
	logger.Debug
	lastAction      int
	NumberOfActions int `json:"numberOfActions"`
}

func init() {
	registry.AddAgent("example-agent", NewAgent)
}

func NewAgent(logger logger.Debug) (rlglue.Agent, error) {
	return &Agent{Debug: logger}, nil
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *Agent) Initialize(expAttr, envAttr rlglue.Attributes) error {
	var ss struct{ Seed int64 }
	err := json.Unmarshal(expAttr, &ss)
	if err != nil {
		agent.Message("warning", "example.Agent seed wasn't available: "+err.Error())
		ss.Seed = 0
	}
	rand.Seed(ss.Seed)

	err = json.Unmarshal(envAttr, &agent)
	if err != nil {
		agent.Message("err", "example.Agent number of Actions wasn't available: "+err.Error())
	}
	agent.lastAction = rand.Intn(agent.NumberOfActions)

	agent.Message("msg", "Example Agent Initialize", "seed", ss.Seed, "numberOfActions", agent.NumberOfActions)

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *Agent) Start(state rlglue.State) rlglue.Action {
	return agent.Step(state, 0)
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *Agent) Step(state rlglue.State, reward float64) rlglue.Action {
	agent.lastAction++
	if agent.lastAction > agent.NumberOfActions {
		agent.lastAction = 0
	}
	return rlglue.Action(agent.lastAction)
}

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *Agent) End(state rlglue.State, reward float64) {}
