package example

import (
	"encoding/json"
	"math/rand"

	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

// Agent just iterates through all actions, starting from a random one.
type Agent struct {
	logger          rlglue.Logger
	lastAction      int
	numberOfActions int
}

func init() {
	rlglue.RegisterAgent("example-agent", NewAgent)
}

func NewAgent() (rlglue.Agent, error) {
	return &Agent{}, nil
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *Agent) Initialize(expAttr, envAttr rlglue.Attributes, logger rlglue.Logger) error {
	agent.logger = logger

	var seed int64
	err := json.Unmarshal(expAttr, &seed)
	if err != nil {
		logger.Message("example.Agent seed wasn't available")
		seed = 0
	}
	rand.Seed(seed)

	err = json.Unmarshal(expAttr, &agent.numberOfActions)
	if err != nil {
		logger.Message("example.Agent number of Actions wasn't available")
	}
	agent.lastAction = rand.Intn(agent.numberOfActions)

	logger.Message("Example Agent Initialize", "seed", seed, "numberOfActions", agent.numberOfActions)

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *Agent) Start(state rlglue.State) rlglue.Action {
	return agent.Step(state, 0)
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *Agent) Step(state rlglue.State, reward float64) rlglue.Action {
	agent.lastAction++
	if agent.lastAction > agent.numberOfActions {
		agent.lastAction = 0
	}
	return rlglue.Action(agent.lastAction)
}

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *Agent) End(state rlglue.State, reward float64) {}
