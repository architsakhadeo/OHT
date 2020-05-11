package agent

import (
	"encoding/json"
	"math/rand"

	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
	"github.com/stellentus/cartpoles/go-src/lib/utils/buffer"
	"github.com/stellentus/cartpoles/go-src/lib/utils/network"
)

type Dqn struct {
	logger.Debug
	lastAction          int
	lastState			rlglue.State
	EnableDebug         bool
	NumberOfActions     int `json:"numberOfActions"`
	StateContainsReplay bool `json:"state-contains-replay"`
	gamma				float64 `json:"alpha"`

	state_len			int `json:"state-len"`
	batch_size			int `json:"batch-size"`

	learning_net		network.Vanilla
	target_net			network.Vanilla

	bf					*buffer.Buffer
	bsize				int `json:"buffer-size"`
	btype				string `json:buffer-type`
}

func init() {
	Add("dqn", NewDqn)
}

func NewDqn(logger logger.Debug) (rlglue.Agent, error) {
	return &Dqn{Debug: logger}, nil
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *Dqn) Initialize(expAttr, envAttr rlglue.Attributes) error {
	var ss struct {
		Seed        int64
		EnableDebug bool `json:"enable-debug"`
	}
	err := json.Unmarshal(expAttr, &ss)
	if err != nil {
		agent.Message("warning", "agent.Example seed wasn't available: "+err.Error())
		ss.Seed = 0
	}
	agent.EnableDebug = ss.EnableDebug

	err = json.Unmarshal(envAttr, &agent)
	if err != nil {
		agent.Message("err", "agent.Example number of Actions wasn't available: "+err.Error())
	}

	rng := rand.New(rand.NewSource(ss.Seed)) // Create a new rand source for reproducibility
	agent.lastAction = rng.Intn(agent.NumberOfActions)

	if agent.EnableDebug {
		agent.Message("msg", "agent.Example Initialize", "seed", ss.Seed, "numberOfActions", agent.NumberOfActions)
	}

	// agent.learning_net = network.Vanilla
	// agent.target_net = agent.SyncTarget()


	agent.bf = buffer.NewBuffer()
	agent.bf.Initialize(agent.btype, agent.bsize, agent.state_len)

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *Dqn) Start(state rlglue.State) rlglue.Action {
	if agent.EnableDebug {
		agent.Message("msg", "start")
	}
	act := agent.Step(state, 0)
	return act
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *Dqn) Step(state rlglue.State, reward float64) rlglue.Action {
	agent.Feed(agent.lastState, agent.lastAction, state, reward, agent.gamma)
	agent.Update()
	// agent.lastAction = (agent.lastAction + 1) % agent.NumberOfActions
	agent.lastAction = agent.Policy(state)
	agent.lastState = state
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
func (agent *Dqn) End(state rlglue.State, reward float64) {
	if agent.EnableDebug {
		agent.Message("msg", "end", "state", state, "reward", reward)
	}
}

func (agent *Dqn) Feed(lastS rlglue.State, lastA int, state rlglue.State, reward float64, gamma float64) {
	agent.bf.Feed(lastS, lastA, state, reward, gamma)
}

func (agent *Dqn) Update() {
	samples := agent.bf.Sample(agent.batch_size)

}

// Choose action
func (agent *Dqn) Policy(state rlglue.State) int {
	return 0	
}