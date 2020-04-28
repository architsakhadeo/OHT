package agent

import (
	"encoding/json"
	"math/rand"

	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

// Example just iterates through all actions, starting from a random one.
type Example struct {
	logger.Debug
	lastAction      int
	NumberOfActions int `json:"numberOfActions"`
}

func init() {
	Add("example", NewExample)
}

func NewExample(logger logger.Debug) (rlglue.Agent, error) {
	return &Example{Debug: logger}, nil
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *Example) Initialize(expAttr, envAttr rlglue.Attributes) error {
	var ss struct{ Seed int64 }
	err := json.Unmarshal(expAttr, &ss)
	if err != nil {
		agent.Message("warning", "agent.Example seed wasn't available: "+err.Error())
		ss.Seed = 0
	}

	err = json.Unmarshal(envAttr, &agent)
	if err != nil {
		agent.Message("err", "agent.Example number of Actions wasn't available: "+err.Error())
	}

	rng := rand.New(rand.NewSource(ss.Seed)) // Create a new rand source for reproducibility
	agent.lastAction = rng.Intn(agent.NumberOfActions)

	agent.Message("msg", "agent.Example Initialize", "seed", ss.Seed, "numberOfActions", agent.NumberOfActions)

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *Example) Start(state rlglue.State) rlglue.Action {
	return agent.Step(state, 0)
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *Example) Step(state rlglue.State, reward float64) rlglue.Action {
	agent.lastAction++
	if agent.lastAction > agent.NumberOfActions {
		agent.lastAction = 0
	}
	return rlglue.Action(agent.lastAction)
}

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *Example) End(state rlglue.State, reward float64) {}
