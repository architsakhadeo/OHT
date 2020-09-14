package agent

import (
	"encoding/json"
	"fmt"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

type exampleSettings struct {
	Seed        int64
	EnableDebug bool `json:"enable-debug"`
}

type environmentAttributes struct {
	NumberOfActions     int  `json:"numberOfActions"`
	StateContainsReplay bool `json:"state-contains-replay"`
}

// Example just iterates through all actions, starting from a random one.
type Example struct {
	logger.Debug
	exampleSettings
	environmentAttributes
	lastAction int
}

func init() {
	Add("example", NewExample)
}

func NewExample(logger logger.Debug) (rlglue.Agent, error) {
	return &Example{Debug: logger}, nil
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *Example) Initialize(run uint, expAttr, envAttr rlglue.Attributes) error {
	err := json.Unmarshal(expAttr, &agent.exampleSettings)
	if err != nil {
		agent.Message("warning", "agent.Example settings weren't available: "+err.Error())
	}

	err = json.Unmarshal(envAttr, &agent.environmentAttributes)
	if err != nil {
		agent.Message("err", "agent.Example environment attributes weren't available: "+err.Error())
	}

	rng := rand.New(rand.NewSource(agent.Seed + int64(run))) // Create a new rand source for reproducibility
	agent.lastAction = rng.Intn(agent.NumberOfActions)

	if agent.EnableDebug {
		agent.Message("msg", "agent.Example Initialize", "seed", agent.Seed, "numberOfActions", agent.NumberOfActions)
	}

	agent.Message("agent settings", fmt.Sprintf("%+v", agent.exampleSettings))

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *Example) Start(state rlglue.State) rlglue.Action {
	if agent.EnableDebug {
		agent.Message("msg", "start")
	}
	act := agent.Step(state, 0)
	return act
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *Example) Step(state rlglue.State, reward float64) rlglue.Action {
	agent.lastAction = (agent.lastAction + 1) % agent.NumberOfActions
	act := rlglue.Action(agent.lastAction)
	if agent.EnableDebug {
		if agent.StateContainsReplay {
			agent.Message("msg", "step", "state", state[0], "reward", reward, "action", act, "expected action", state[1])
		} else {
			agent.Message("msg", "step", "state", state, "reward", reward, "action", act)
		}
	}
	return act
}

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *Example) End(state rlglue.State, reward float64) {
	if agent.EnableDebug {
		agent.Message("msg", "end", "state", state, "reward", reward)
	}
}

func (agent *Example) GetLock() bool {
	return false
}