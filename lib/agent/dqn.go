package agent

import (
	"encoding/json"
	"math"
	"math/rand"

	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"github.com/stellentus/cartpoles/lib/util/network"
	"gonum.org/v1/gonum/mat"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	"github.com/stellentus/cartpoles/lib/util/buffer"
)

type Model struct {
}

type Dqn struct {
	logger.Debug
	rng                 *rand.Rand
	lastAction          int
	lastState           rlglue.State
	EnableDebug         bool
	NumberOfActions     int     `json:"numberOfActions"`
	StateContainsReplay bool    `json:"state-contains-replay"`
	Gamma               float64 `json:"gamma"`
	Epsilon             float64 `json:"epsilon"`
	Hidden              []int   `json:"dqn-hidden"`
	Alpha               float64 `json:"alpha"`
	Sync                int     `json:"dqn-sync"`
	Decay               float64 `json:"dqn-decay"`
	updateNum           int

	bf    *buffer.Buffer
	Bsize int    `json:"buffer-size"`
	Btype string `json:"buffer-type"`

	StateDim  int `json:"state-len"`
	BatchSize int `json:"dqn-batch"`

	StateRange []float64

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
	var ss struct {
		Seed        int64
		EnableDebug bool `json:"enable-debug"`

		NumberOfActions     int     `json:"numberOfActions"`
		StateContainsReplay bool    `json:"state-contains-replay"`
		Gamma               float64 `json:"gamma"`
		Epsilon             float64 `json:"epsilon"`
		Hidden              []int   `json:"dqn-hidden"`
		Alpha               float64 `json:"alpha"`
		Sync                int     `json:"dqn-sync"`
		Decay               float64 `json:"dqn-decay"`

		Bsize int    `json:"buffer-size"`
		Btype string `json:"buffer-type"`

		StateDim  int `json:"state-len"`
		BatchSize int `json:"dqn-batch"`

		StateRange []float64 `json:"StateRange"`
	}
	err := json.Unmarshal(expAttr, &ss)
	if err != nil {
		agent.Message("warning", "agent.Example seed wasn't available: "+err.Error())
		ss.Seed = 0
	}
	agent.EnableDebug = ss.EnableDebug
	agent.NumberOfActions = ss.NumberOfActions
	agent.StateContainsReplay = ss.StateContainsReplay
	agent.Gamma = ss.Gamma
	agent.Epsilon = ss.Epsilon
	agent.Hidden = ss.Hidden
	agent.Alpha = ss.Alpha
	agent.Sync = ss.Sync
	agent.Bsize = ss.Bsize
	agent.Btype = ss.Btype
	agent.StateDim = ss.StateDim
	agent.BatchSize = ss.BatchSize
	agent.StateRange = ss.StateRange
	agent.Decay = ss.Decay

	err = json.Unmarshal(envAttr, &agent)

	if err != nil {
		agent.Message("err", "agent.Example number of Actions wasn't available: "+err.Error())
	}
	// agent.rng = rand.New(rand.NewSource(ss.Seed)) // Create a new rand source for reproducibility
	// agent.lastAction = rng.Intn(agent.NumberOfActions)
	agent.rng = rand.New(rand.NewSource(ss.Seed + int64(run))) // Create a new rand source for reproducibility

	if agent.EnableDebug {
		agent.Message("msg", "agent.Example Initialize", "seed", ss.Seed, "numberOfActions", agent.NumberOfActions)
	}
	agent.bf = buffer.NewBuffer()
	agent.bf.Initialize(agent.Btype, agent.Bsize, agent.StateDim)

	// NN: Graph Construction
	// NN: Weight Initialization
	agent.learningNet = network.CreateNetwork(agent.StateDim, agent.Hidden, agent.NumberOfActions, agent.Alpha, agent.Decay)
	agent.targetNet = network.CreateNetwork(agent.StateDim, agent.Hidden, agent.NumberOfActions, agent.Alpha, agent.Decay)
	agent.updateNum = 0

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *Dqn) Start(state rlglue.State) rlglue.Action {
	state = agent.StateNormalization(state)

	if agent.EnableDebug {
		agent.Message("msg", "start")
	}
	agent.lastState = state
	act := agent.Policy(state)
	agent.lastAction = act
	return rlglue.Action(act)
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *Dqn) Step(state rlglue.State, reward float64) rlglue.Action {

	state = agent.StateNormalization(state)
	agent.Feed(agent.lastState, agent.lastAction, state, reward, agent.Gamma)
	agent.Update()
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
		copy(agent.targetNet.HiddenWeights, agent.learningNet.HiddenWeights)
		agent.targetNet.OutputWeights = agent.learningNet.OutputWeights
	}
	// fmt.Println(mat.Row(nil, 0, agent.targetNet.OutputWeights)[0:3])

	samples := agent.bf.Sample(agent.BatchSize)
	lastStates := ao.Index2d(samples, 0, len(samples), 0, agent.StateDim)
	lastActions := ao.Flatten2DInt(ao.A64ToInt2D(ao.Index2d(samples, 0, len(samples), agent.StateDim, agent.StateDim+1)))
	states := ao.Index2d(samples, 0, len(samples), agent.StateDim+1, agent.StateDim*2+1)
	rewards := ao.Flatten2DFloat(ao.Index2d(samples, 0, len(samples), agent.StateDim*2+1, agent.StateDim*2+2))
	gammas := ao.Flatten2DFloat(ao.Index2d(samples, 0, len(samples), agent.StateDim*2+2, agent.StateDim*2+3))

	// NN: Weight update
	lastQMat := agent.learningNet.Forward(lastStates)
	var lastQ [][]float64
	for i := 0; i < len(lastStates); i++ {
		lastQ = append(lastQ, mat.Row(nil, i, lastQMat))
	}
	lastActionValue := ao.RowIndexFloat(lastQ, lastActions)
	targetQMat := agent.learningNet.Predict(states)
	var targetQ [][]float64
	for i := 0; i < len(states); i++ {
		targetQ = append(targetQ, mat.Row(nil, i, targetQMat))
	}
	targetActionValue := ao.RowIndexMax(targetQ)

	loss := float64(0)
	for i := 0; i < len(lastQ); i++ {
		loss = loss + math.Pow(rewards[i]+gammas[i]*targetActionValue[i]-lastActionValue[i], 2)
	}
	loss = loss / float64(len(lastQ)) / 2.0
	agent.learningNet.Backward(loss)
	agent.updateNum += 1
}

// Choose action
func (agent *Dqn) Policy(state rlglue.State) int {
	var idx int
	if rand.Float64() < agent.Epsilon {
		idx = agent.rng.Intn(agent.NumberOfActions)
	} else {
		// NN: choose action
		var inputS [][]float64 //rlglue.State
		inputS = append(inputS, agent.lastState)
		allValueMat := agent.learningNet.Predict(inputS)
		allValue := mat.Row(nil, 0, allValueMat)
		var argmax int
		max := math.Inf(-1)
		for i := 0; i < agent.NumberOfActions; i++ {
			v := allValue[i]
			if v > max {
				max = v
				argmax = i
			}
		}
		idx = argmax
	}
	return idx
}
