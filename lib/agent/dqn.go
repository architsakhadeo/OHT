package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"

	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"github.com/stellentus/cartpoles/lib/util/network"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	"github.com/stellentus/cartpoles/lib/util/buffer"
)

type dqnSettings struct {
	Seed        int64
	EnableDebug bool `json:"enable-debug"`

	NumberOfActions     int     `json:"numberOfActions"`
	StateContainsReplay bool    `json:"state-contains-replay"`
	Gamma               float64 `json:"gamma"`
	Epsilon             float64 `json:"epsilon"`
	MinEpsilon          float64 `json:"min-epsilon"`
	DecreasingEpsilon   string  `json:"decreasing-epsilon"`

	Hidden    []int   `json:"dqn-hidden"`
	Alpha     float64 `json:"alpha"`
	Sync      int     `json:"dqn-sync"`
	Decay     float64 `json:"dqn-decay"`
	Momentum  float64 `json:"dqn-momentum"`
	AdamBeta1 float64 `json:"dqn-adamBeta1"`
	AdamBeta2 float64 `json:"dqn-adamBeta2"`
	AdamEps   float64 `json:"dqn-adamEps"`

	Bsize int    `json:"buffer-size"`
	Btype string `json:"buffer-type"`

	StateDim  int `json:"state-len"`
	BatchSize int `json:"dqn-batch"`

	StateRange []float64 `json:"StateRange"`
}

type Dqn struct {
	logger.Debug
	rng        *rand.Rand
	lastAction int
	lastState  rlglue.State

	dqnSettings

	updateNum int
	learning  bool
	stepNum   int

	bf *buffer.Buffer

	learningNet network.Network
	targetNet   network.Network
}

func init() {
	Add("dqn", NewDqn)
}

func NewDqn(logger logger.Debug) (rlglue.Agent, error) {
	return &Dqn{Debug: logger}, nil
}

func (agent *Dqn) Initialize(run uint, expAttr, envAttr rlglue.Attributes) error {
	err := json.Unmarshal(expAttr, &agent.dqnSettings)
	if err != nil {
		return errors.New("DQN agent attributes were not valid: " + err.Error())
	}

	if agent.DecreasingEpsilon == "None" {
		agent.MinEpsilon = 0 // Not used
	} else {
		agent.Epsilon = 1.0
	}

	agent.learning = false
	agent.stepNum = 0

	err = json.Unmarshal(envAttr, &agent)

	if err != nil {
		agent.Message("err", "agent.Example number of Actions wasn't available: "+err.Error())
	}
	agent.rng = rand.New(rand.NewSource(agent.Seed + int64(run))) // Create a new rand source for reproducibility

	if agent.EnableDebug {
		agent.Message("msg", "agent.Example Initialize", "seed", agent.Seed, "numberOfActions", agent.NumberOfActions)
	}
	agent.bf = buffer.NewBuffer()
	agent.bf.Initialize(agent.Btype, agent.Bsize, agent.StateDim, agent.Seed+int64(run))

	// NN: Graph Construction
	// NN: Weight Initialization
	agent.learningNet = network.CreateNetwork(agent.StateDim, agent.Hidden, agent.NumberOfActions, agent.Alpha,
		agent.Decay, agent.Momentum, agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps)
	agent.targetNet = network.CreateNetwork(agent.StateDim, agent.Hidden, agent.NumberOfActions, agent.Alpha,
		agent.Decay, agent.Momentum, agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps)
	agent.updateNum = 0

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *Dqn) Start(state rlglue.State) rlglue.Action {

	state = agent.StateNormalization(state)
	agent.lastState = state
	act := agent.Policy(state)
	agent.lastAction = act

	if agent.EnableDebug {
		agent.Message("msg", "start")
	}
	return rlglue.Action(act)
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *Dqn) Step(state rlglue.State, reward float64) rlglue.Action {
	if reward != 0 {
		agent.learning = true
	}
	if agent.DecreasingEpsilon == "step" {
		agent.Epsilon = math.Max(agent.Epsilon-1.0/10000, agent.MinEpsilon)
		fmt.Println(agent.Epsilon)
	}

	state = agent.StateNormalization(state)
	agent.Feed(agent.lastState, agent.lastAction, state, reward, agent.Gamma)
	agent.stepNum = agent.stepNum + 1
	agent.Update()
	agent.lastState = state
	agent.lastAction = agent.Policy(state)
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
	agent.Feed(agent.lastState, agent.lastAction, state, reward, float64(0))
	if agent.EnableDebug {
		agent.Message("msg", "end", "state", state, "reward", reward)
	}
}

func (agent *Dqn) StateNormalization(state rlglue.State) rlglue.State {
	for i := 0; i < agent.StateDim; i++ {
		state[i] = state[i] / agent.StateRange[i]
	}
	return state
}

func (agent *Dqn) Feed(lastS rlglue.State, lastA int, state rlglue.State, reward float64, gamma float64) {
	agent.bf.Feed(lastS, lastA, state, reward, gamma)
}

func (agent *Dqn) Update() {

	if agent.updateNum%agent.Sync == 0 {
		// NN: Synchronization
		for i := 0; i < len(agent.targetNet.HiddenWeights); i++ {
			agent.targetNet.HiddenWeights[i] = agent.learningNet.HiddenWeights[i]
		}
		agent.targetNet.OutputWeights = agent.learningNet.OutputWeights

	}

	samples := agent.bf.Sample(agent.BatchSize)
	lastStates := ao.Index2d(samples, 0, len(samples), 0, agent.StateDim)
	lastActions := ao.Flatten2DInt(ao.A64ToInt2D(ao.Index2d(samples, 0, len(samples), agent.StateDim, agent.StateDim+1)))
	states := ao.Index2d(samples, 0, len(samples), agent.StateDim+1, agent.StateDim*2+1)
	rewards := ao.Flatten2DFloat(ao.Index2d(samples, 0, len(samples), agent.StateDim*2+1, agent.StateDim*2+2))
	gammas := ao.Flatten2DFloat(ao.Index2d(samples, 0, len(samples), agent.StateDim*2+2, agent.StateDim*2+3))

	// NN: Weight update
	lastQ := agent.learningNet.Forward(lastStates)
	lastActionValue := ao.RowIndexFloat(lastQ, lastActions)
	targetQ := agent.learningNet.Predict(states)
	targetActionValue := ao.RowIndexMax(targetQ)

	loss := make([][]float64, len(lastQ))
	for i := 0; i < len(lastQ); i++ {
		loss[i] = make([]float64, agent.NumberOfActions)
	}
	for i := 0; i < len(lastQ); i++ {
		for j := 0; j < agent.NumberOfActions; j++ {
			loss[i][j] = 0
		}
		loss[i][lastActions[i]] = rewards[i] + gammas[i]*targetActionValue[i] - lastActionValue[i]
	}

	agent.learningNet.Backward(loss)
	agent.updateNum += 1
}

// Choose action
func (agent *Dqn) Policy(state rlglue.State) int {
	var idx int
	if (agent.rng.Float64() < agent.Epsilon) || (!agent.learning) {
		idx = agent.rng.Intn(agent.NumberOfActions)
	} else {
		// NN: choose action
		inputS := make([][]float64, 1)
		inputS[0] = agent.lastState
		allValue := agent.learningNet.Predict(inputS)[0]
		var argmax int
		var v float64
		max := math.Inf(-1)
		for i := 0; i < len(allValue); i++ {
			v = allValue[i]
			if v > max {
				max = v
				argmax = i
			}
		}
		idx = argmax
	}
	return idx
}
