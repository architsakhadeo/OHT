package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"github.com/stellentus/cartpoles/lib/util/buffer"
	"github.com/stellentus/cartpoles/lib/util/lockweight"
	"github.com/stellentus/cartpoles/lib/util/network"
	"github.com/stellentus/cartpoles/lib/util/normalizer"
	"github.com/stellentus/cartpoles/lib/util/optimizer"
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

	StateDim   int  `json:"state-len"`
	BatchSize  int  `json:"dqn-batch"`
	IncreaseBS bool `json:"increasing-batch"`

	StateRange []float64 `json:"StateRange"`

	OptName string `json:"optimizer"`
}

type Dqn struct {
	logger.Debug
	rng        *rand.Rand
	lastAction int
	lastState  rlglue.State

	dqnSettings

	updateNum int
	learning  bool
	//stepNum   int

	nml normalizer.Normalizer
	bf  *buffer.Buffer

	learningNet network.Network
	targetNet   network.Network
	opt         optimizer.Optimizer

	lw   lockweight.LockWeight
	lock bool
}

func init() {
	Add("dqn", NewDqn)
}

func NewDqn(logger logger.Debug) (rlglue.Agent, error) {
	return &Dqn{Debug: logger}, nil
}

func (agent *Dqn) InitLockWeight(lw lockweight.LockWeight) lockweight.LockWeight {
	lw.DecCount = 0
	lw.BestAvg = math.Inf(-1)

	if lw.LockCondition == "dynamic-reward" {
		lw.CheckChange = agent.DynamicLock
	} else if lw.LockCondition == "onetime-reward" {
		lw.CheckChange = agent.OnetimeRwdLock
	} else if lw.LockCondition == "onetime-epLength" {
		lw.CheckChange = agent.OnetimeEpLenLock
	} else if lw.LockCondition == "beginning" {
		lw.CheckChange = agent.KeepLock
	}
	return lw
}

func (agent *Dqn) Initialize(run uint, expAttr, envAttr rlglue.Attributes, sweepIdx int) error {
	err := json.Unmarshal(expAttr, &agent.dqnSettings)
	if err != nil {
		return errors.New("DQN agent attributes were not valid: " + err.Error())
	}

	err = json.Unmarshal(expAttr, &agent.lw)
	if err != nil {
		return errors.New("DQN agent LockWeight attributes were not valid: " + err.Error())
	}
	agent.lw = agent.InitLockWeight(agent.lw)

	if agent.DecreasingEpsilon == "None" {
		agent.MinEpsilon = 0 // Not used
	} else {
		agent.Epsilon = 1.0
	}

	agent.learning = false
	//agent.stepNum = 0

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

	agent.nml = normalizer.Normalizer{agent.StateDim, agent.StateRange}

	// NN: Graph Construction
	// NN: Weight Initialization
	agent.learningNet = network.CreateNetwork(agent.StateDim, agent.Hidden, agent.NumberOfActions, agent.Alpha,
		agent.Decay, agent.Momentum, agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps)
	agent.targetNet = network.CreateNetwork(agent.StateDim, agent.Hidden, agent.NumberOfActions, agent.Alpha,
		agent.Decay, agent.Momentum, agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps)
	agent.updateNum = 0

	if agent.OptName == "Adam" {
		agent.opt = new(optimizer.Adam)
		agent.opt.Init(agent.Alpha, []float64{agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps}, agent.StateDim, agent.Hidden, agent.NumberOfActions)
	} else if agent.OptName == "Sgd" {
		agent.opt = new(optimizer.Sgd)
		agent.opt.Init(agent.Alpha, []float64{agent.Momentum}, agent.StateDim, agent.Hidden, agent.NumberOfActions)
	} else {
		errors.New("Optimizer NotImplemented")
	}

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
//func (agent *Dqn) Start(state rlglue.State) rlglue.Action {
func (agent *Dqn) Start(oristate rlglue.State) rlglue.Action {
	state := make([]float64, agent.StateDim)
	copy(state, oristate)

	state = agent.nml.MeanZeroNormalization(state)
	agent.lastState = state
	act := agent.Policy(state)
	agent.lastAction = act

	if agent.EnableDebug {
		agent.Message("msg", "start")
	}
	return rlglue.Action(act)
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
//func (agent *Dqn) Step(state rlglue.State, reward float64) rlglue.Action {
func (agent *Dqn) Step(oristate rlglue.State, reward float64) rlglue.Action {
	if reward != 0 {
		agent.learning = true
	}
	if agent.DecreasingEpsilon == "step" {
		agent.Epsilon = math.Max(agent.Epsilon-1.0/10000, agent.MinEpsilon)
		fmt.Println(agent.Epsilon)
	}
	state := make([]float64, agent.StateDim)
	copy(state, oristate)
	state = agent.nml.MeanZeroNormalization(state)
	agent.bf.Feed(agent.lastState, agent.lastAction, state, reward, agent.Gamma)
	//agent.stepNum = agent.stepNum + 1
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
	agent.bf.Feed(agent.lastState, agent.lastAction, state, reward, float64(0)) // gamma=0
	agent.Update()
	if agent.EnableDebug {
		agent.Message("msg", "end", "state", state, "reward", reward)
	}
}

func (agent *Dqn) Update() {
	if agent.lw.UseLock {
		//if agent.updateNum%agent.Bsize == 0 {
		//	agent.lock = agent.lw.CheckChange()
		//}
		agent.lock = agent.lw.CheckChange()
		if agent.lock {
			agent.updateNum += 1
			return
		}
	}

	if agent.IncreaseBS {
		if agent.updateNum%1000 == 0 {
			//inc := int(math.Min(float64(agent.BatchSize) * agent.Alpha, 1.0))
			inc := int(1 / (1 - float64(agent.AdamBeta1)))
			agent.BatchSize = int(math.Min(float64(agent.BatchSize+inc),
				float64(agent.Bsize/2)))
			fmt.Println("batch size increase to", agent.BatchSize)
		}
	}

	if agent.updateNum%agent.Sync == 0 {
		// NN: Synchronization
		//for i := 0; i < len(agent.targetNet.HiddenWeights); i++ {
		//	agent.targetNet.HiddenWeights[i] = agent.learningNet.HiddenWeights[i]
		//}
		//agent.targetNet.OutputWeights = agent.learningNet.OutputWeights
		////fmt.Println("sync", agent.updateNum)
		agent.targetNet = network.Synchronization(agent.learningNet, agent.targetNet)
	}

	lastStates, lastActionsFloat, states, rewards, gammas := agent.bf.Sample(agent.BatchSize)
	lastActions := ao.Flatten2DInt(ao.A64ToInt2D(lastActionsFloat))

	// NN: Weight update
	lastQ := agent.learningNet.Forward(lastStates)
	lastActionValue := ao.RowIndexFloat(lastQ, lastActions)
	targetQ := agent.targetNet.Predict(states)
	targetActionValue, _ := ao.RowIndexMax(targetQ)

	loss := make([][]float64, len(lastQ))
	for i := 0; i < len(lastQ); i++ {
		loss[i] = make([]float64, agent.NumberOfActions)
	}
	for i := 0; i < len(lastQ); i++ {
		for j := 0; j < agent.NumberOfActions; j++ {
			loss[i][j] = 0
		}
		loss[i][lastActions[i]] = rewards[i][0] + gammas[i][0]*targetActionValue[i] - lastActionValue[i]
	}
	avgLoss := make([][]float64, 1)
	avgLoss[0] = make([]float64, agent.NumberOfActions)
	for j := 0; j < agent.NumberOfActions; j++ {
		sum := 0.0
		for i := 0; i < len(loss); i++ {
			sum += loss[i][j]
		}
		avgLoss[0][j] = sum / float64(len(loss))
	}

	agent.learningNet.Backward(loss, agent.opt)
	//agent.learningNet.Backward(avgLoss)
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

func (agent *Dqn) CheckAvgRwdLock(avg float64) bool {
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

func (agent *Dqn) CheckAvgRwdUnlock(avg float64) bool {
	if agent.lw.LockAvgRwd > avg {
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

//func (agent *Dqn) CheckChange() bool {
func (agent *Dqn) DynamicLock() bool {
	_, _, _, rewards2D, _ := agent.bf.Content()
	rewards := ao.Flatten2DFloat(rewards2D)
	avg := ao.Average(rewards)
	if len(rewards) < agent.Bsize {
		return false
	}
	if agent.lock {
		lock := agent.CheckAvgRwdUnlock(avg)
		if !lock {
			agent.lw.LockAvgRwd = avg
			agent.lw.DecCount = 0
			fmt.Println("UnLocked")
		}
		return lock
	} else {
		lock := agent.CheckAvgRwdLock(avg)
		if lock {
			agent.lw.LockAvgRwd = avg
			agent.lw.DecCount = 0
			fmt.Println("Locked")
		}
		return lock
	}
}

func (agent *Dqn) OnetimeRwdLock() bool {
	if agent.lock {
		return true
	} else {
		_, _, _, rewards2D, _ := agent.bf.Content()
		rewards := ao.Flatten2DFloat(rewards2D)
		avg := ao.Average(rewards)
		if len(rewards) < agent.Bsize {
			return false
		}
		if avg > agent.lw.LockAvgRwd {
			return true
		}
		return false
	}
}

func (agent *Dqn) OnetimeEpLenLock() bool {
	if agent.lock {
		return true
	} else {
		_, _, _, rewards2D, _ := agent.bf.Content()
		rewards := ao.Flatten2DFloat(rewards2D)
		if len(rewards) < agent.Bsize {
			return false
		}
		zeros := 0
		for i := 0; i < len(rewards); i++ {
			if rewards[i] == 0 {
				zeros += 1
			}
		}
		if zeros != 0 {
			avg := float64(agent.Bsize) / float64(zeros)
			if avg < agent.lw.LockAvgLen {
				return true
			}
		}
		return false
	}
}

func (agent *Dqn) KeepLock() bool {
	return true
}

func (agent *Dqn) GetLock() bool {
	return agent.lock
}

func (agent *Dqn) SaveWeights(basePath string) error {
	return nil
}

func (agent *Dqn) GetLearnProg() string {
	return "0"
}
