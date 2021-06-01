package agent

import (
	"encoding/json"
	"fmt"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

type randomSettings struct {
	Seed        int64
	EnableDebug bool `json:"enable-debug"`
}

// Random takes a random action at each timestep.
type Random struct {
	logger.Debug
	randomSettings
	environmentAttributes
	rng *rand.Rand
}

func init() {
	Add("random", NewRandom)
}

func NewRandom(logger logger.Debug) (rlglue.Agent, error) {
	return &Random{Debug: logger}, nil
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *Random) Initialize(run uint, expAttr, envAttr rlglue.Attributes, sweepIdx int) error {
	err := json.Unmarshal(expAttr, &agent.randomSettings)
	if err != nil {
		agent.Message("warning", "agent.Random settings weren't available: "+err.Error())
	}

	err = json.Unmarshal(envAttr, &agent.environmentAttributes)
	if err != nil {
		agent.Message("err", "agent.Random environment attributes weren't available: "+err.Error())
	}

	agent.rng = rand.New(rand.NewSource(agent.Seed + int64(run))) // Create a new rand source for reproducibility

	if agent.EnableDebug {
		agent.Message("msg", "agent.Random Initialize", "seed", agent.Seed, "numberOfActions", agent.NumberOfActions)
	}

	agent.Message("agent settings", fmt.Sprintf("%+v", agent.randomSettings))

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *Random) Start(state rlglue.State) rlglue.Action {
	if agent.EnableDebug {
		agent.Message("msg", "start")
	}
	act := agent.Step(state, 0)
	return act
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *Random) Step(state rlglue.State, reward float64) rlglue.Action {
	act := rlglue.Action(agent.rng.Intn(agent.NumberOfActions))
	if agent.EnableDebug {
		agent.Message("msg", "step", "state", state, "reward", reward, "action", act)
	}
	return act
}

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *Random) End(state rlglue.State, reward float64) {
	if agent.EnableDebug {
		agent.Message("msg", "end", "state", state, "reward", reward)
	}
}

func (agent *Random) GetLock() bool {
	return false
}

func (agent *Random) SaveWeights(basePath string) error {
	return nil
}

func (agent *Random) GetLearnProg() string {
	return "0"
}
