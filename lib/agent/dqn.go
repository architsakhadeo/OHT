package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"github.com/stellentus/cartpoles/lib/util/network"
	"math"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	"github.com/stellentus/cartpoles/lib/util/buffer"
)

type LockWeight struct {
	UseLock		bool
	DecCount	int
	BestAvg 	float64
	LockAvg		float64
	LockThrd	int
}
func NewLockWeight() LockWeight {
	lw := LockWeight{}
	lw.DecCount = 0
	lw.BestAvg = math.Inf(-1)
	lw.LockThrd = 2
	lw.UseLock = false
	return lw
}

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

	lw          LockWeight
	lock 		bool
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

		UseLock   bool    `json:"lock-weight"`
	}
	err := json.Unmarshal(expAttr, &ss)
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

	agent.lw = NewLockWeight()
	agent.lock = false
	agent.lw.UseLock = ss.UseLock

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
	if agent.lw.UseLock {
		if agent.updateNum%agent.Bsize == 0 {
			agent.lock = agent.CheckChange()
		}
		if agent.lock {
			agent.updateNum += 1
			return
		}
	}

	if agent.updateNum%agent.Sync == 0 {
		// NN: Synchronization
		for i := 0; i < len(agent.targetNet.HiddenWeights); i++ {
			agent.targetNet.HiddenWeights[i] = agent.learningNet.HiddenWeights[i]
		}
		agent.targetNet.OutputWeights = agent.learningNet.OutputWeights
	}

	lastStates, lastActions, states, rewards, gammas := agent.bf.Sample(agent.BatchSize)

	// NN: Weight update
	lastQ := agent.learningNet.Forward(lastStates)
	lastActionValue := ao.RowIndexFloat(lastQ, lastActions)
	targetQ := agent.learningNet.Predict(states)
	targetActionValue, _ := ao.RowIndexMax(targetQ)

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
	//fmt.Println(loss[len(lastQ)-1][lastActions[len(lastQ)-1]])

	//temp := 0.0
	//for i := 0; i < len(lastQ); i++ {
	//	temp = temp + (rewards[i] + gammas[i]*targetActionValue[i] - lastActionValue[i])
	//}
	//temp = temp / float64(len(lastQ))
	//for i := 0; i < len(lastQ); i++ {
	//	for j := 0; j < agent.NumberOfActions; j++ {
	//		if j != lastActions[i] {
	//			loss[i][j] = 0
	//		} else {
	//			loss[i][j] = temp
	//		}
	//	}
	//}
	//fmt.Println(loss[len(lastQ)-1][lastActions[len(lastQ)-1]])

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
		inputS[0] = state

		allValue := agent.learningNet.Predict(inputS)
		_, idxs := ao.RowIndexMax(allValue)
		idx = idxs[0]

	}
	return idx
}

func (agent *Dqn) CheckLock(avg float64) bool {
	if agent.lw.BestAvg > avg {
		agent.lw.DecCount += 1
		fmt.Println("Count to lock", agent.lw.DecCount)
	} else {
		agent.lw.BestAvg = avg
		agent.lw.DecCount = 0
	}
	var lock bool
	if agent.lw.DecCount > agent.lw.LockThrd {
		agent.lw.DecCount = 0
		lock = true
	} else {
		lock = false
	}
	return lock
}

func (agent *Dqn) CheckUnlock(avg float64) bool {
	if agent.lw.LockAvg > avg {
		agent.lw.DecCount += 1
		fmt.Println("Count to unlock", agent.lw.DecCount)
	} else {
		agent.lw.DecCount = 0
	}
	var lock bool
	if agent.lw.DecCount > agent.lw.LockThrd {
		agent.lw.DecCount = 0
		lock = false
	} else {
		lock = true
	}
	return lock
}

func (agent *Dqn) CheckChange() bool {
	_, _, _, rewards, _ := agent.bf.Content()
	avg := ao.Average(rewards)
	if len(rewards) < agent.Bsize {
		return false
	}
	if agent.lock {
		lock := agent.CheckUnlock(avg)
		if !lock {
			agent.lw.LockAvg = avg
			agent.lw.DecCount = 0
			fmt.Println("UnLocked")
		}
		return lock
	} else {
		lock := agent.CheckLock(avg)
		if lock {
			agent.lw.LockAvg = avg
			agent.lw.DecCount = 0
			fmt.Println("Locked")
		}
		return lock
	}
}
