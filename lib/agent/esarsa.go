package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"

	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"github.com/stellentus/cartpoles/lib/util/buffer"
	"github.com/stellentus/cartpoles/lib/util/lockweight"

	tpo "github.com/stellentus/cartpoles/lib/util/type-opr"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	"github.com/stellentus/cartpoles/lib/util"
)

const (
	maxPosition        = 2.4
	maxVelocity        = 4
	maxAngle           = 12 * 2 * math.Pi / 360
	maxAngularVelocity = 3.5
)

type esarsaSettings struct {
	EnableDebug        bool    `json:"enable-debug"`
	Seed               int64   `json:"seed"`
	NumTilings         int     `json:"tilings"`
	NumTiles           int     `json:"tiles"`
	Gamma              float64 `json:"gamma"`
	Lambda             float64 `json:"lambda"`
	Epsilon            float64 `json:"epsilon"`
	Alpha              float64 `json:"alpha"`
	AdaptiveAlpha      float64 `json:"adaptive-alpha"`
	IsStepsizeAdaptive bool    `json:"is-stepsize-adaptive"`

	StateDim int    `json:"state-len"`
	Bsize    int    `json:"buffer-size"`
	Btype    string `json:"buffer-type"`
}

// Expected sarsa-lambda with tile coding
type ESarsa struct {
	logger.Debug
	rng   *rand.Rand
	tiler util.MultiTiler

	// Agent accessible parameters
	weights                [][]float64 // weights is a slice of weights for each action
	traces                 [][]float64
	delta                  float64
	oldState               rlglue.State
	oldStateActiveFeatures []int
	oldAction              rlglue.Action
	stepsize               float64
	beta1                  float64
	beta2                  float64
	e                      float64
	m                      [][]float64
	v                      [][]float64
	timesteps              float64
	accumulatingbeta1      float64
	accumulatingbeta2      float64
	esarsaSettings

	bf   *buffer.Buffer
	lw   lockweight.LockWeight
	lock bool
}

func init() {
	Add("esarsa", NewESarsa)
}

func NewESarsa(logger logger.Debug) (rlglue.Agent, error) {
	return &ESarsa{Debug: logger}, nil
}

func (agent *ESarsa) InitLockWeight(lw lockweight.LockWeight) lockweight.LockWeight {

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

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *ESarsa) Initialize(run uint, expAttr, envAttr rlglue.Attributes) error {
	agent.esarsaSettings = esarsaSettings{
		// These default settings will be used if the config doesn't set these values
		NumTilings:         32,
		NumTiles:           4,
		Gamma:              0.99,
		Lambda:             0.8,
		Epsilon:            0.05,
		Alpha:              0.1,
		AdaptiveAlpha:      0.001,
		IsStepsizeAdaptive: false,
	}

	err := json.Unmarshal(expAttr, &agent.esarsaSettings)
	if err != nil {
		agent.Message("warning", "agent.ESarsa settings weren't available: "+err.Error())
		agent.esarsaSettings.Seed = 0
	}
	err = json.Unmarshal(expAttr, &agent.lw)
	if err != nil {
		return errors.New("ESarsa agent LockWeight attributes were not valid: " + err.Error())
	}
	agent.bf = buffer.NewBuffer()
	agent.bf.Initialize(agent.Btype, agent.Bsize, agent.StateDim, agent.Seed+int64(run))
	agent.lw = agent.InitLockWeight(agent.lw)

	if agent.IsStepsizeAdaptive == false {
		agent.stepsize = agent.Alpha / float64(agent.esarsaSettings.NumTilings) // Setting stepsize
	} else {
		agent.stepsize = agent.AdaptiveAlpha / float64(agent.esarsaSettings.NumTilings) // Setting adaptive stepsize
	}

	agent.beta1 = 0.9
	agent.beta2 = 0.999
	agent.e = math.Pow(10, -8)

	agent.esarsaSettings.Seed += int64(run)
	agent.rng = rand.New(rand.NewSource(agent.esarsaSettings.Seed)) // Create a new rand source for reproducibility

	// scales the input observations for tile-coding
	scalers := []util.Scaler{
		util.NewScaler(-maxPosition, maxPosition, agent.esarsaSettings.NumTiles),
		util.NewScaler(-maxVelocity, maxVelocity, agent.esarsaSettings.NumTiles),
		util.NewScaler(-maxAngle, maxAngle, agent.esarsaSettings.NumTiles),
		util.NewScaler(-maxAngularVelocity, maxAngularVelocity, agent.esarsaSettings.NumTiles),
	}

	agent.tiler, err = util.NewMultiTiler(4, agent.esarsaSettings.NumTilings, scalers)
	if err != nil {
		return err
	}

	agent.weights = make([][]float64, 2) // one weight slice for each action
	agent.weights[0] = make([]float64, agent.tiler.NumberOfIndices())
	agent.weights[1] = make([]float64, agent.tiler.NumberOfIndices())

	agent.traces = make([][]float64, 2) // one trace slice for each action
	agent.traces[0] = make([]float64, agent.tiler.NumberOfIndices())
	agent.traces[1] = make([]float64, agent.tiler.NumberOfIndices())

	agent.m = make([][]float64, 2)
	agent.m[0] = make([]float64, agent.tiler.NumberOfIndices())
	agent.m[1] = make([]float64, agent.tiler.NumberOfIndices())

	agent.v = make([][]float64, 2)
	agent.v[0] = make([]float64, agent.tiler.NumberOfIndices())
	agent.v[1] = make([]float64, agent.tiler.NumberOfIndices())

	agent.timesteps = 0

	agent.Message("esarsa settings", fmt.Sprintf("%+v", agent.esarsaSettings))

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *ESarsa) Start(state rlglue.State) rlglue.Action {

	var err error
	agent.oldStateActiveFeatures, err = agent.tiler.Tile(state) // Indices of active features of the tile-coded state

	if err != nil {
		agent.Message("err", "agent.ESarsa is acting on garbage state because it couldn't create tiles: "+err.Error())
	}

	oldA, _ := agent.PolicyExpectedSarsaLambda(agent.oldStateActiveFeatures) // Exp-Sarsa-L policy
	agent.oldAction, _ = tpo.GetInt(oldA)

	agent.timesteps++

	if agent.EnableDebug {
		agent.Message("msg", "start")
	}

	return agent.oldAction

}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *ESarsa) Step(state rlglue.State, reward float64) rlglue.Action {
	newStateActiveFeatures, err := agent.tiler.Tile(state) // Indices of active features of the tile-coded state

	if err != nil {
		agent.Message("err", "agent.ESarsa is acting on garbage state because it couldn't create tiles: "+err.Error())
	}

	agent.delta = reward // TD error calculation begins

	for _, value := range agent.oldStateActiveFeatures {
		oldA, _ := tpo.GetInt(agent.oldAction)
		agent.delta -= agent.weights[oldA][value] // TD error prediction calculation
		agent.traces[oldA][value] = 1             // replacing active traces to 1
	}

	newAction, epsilons := agent.PolicyExpectedSarsaLambda(newStateActiveFeatures) // Exp-Sarsa-L policy

	if agent.lw.UseLock {
		agent.bf.Feed(agent.oldState, agent.oldAction, state, reward, agent.Gamma)
		agent.lock = agent.lw.CheckChange()
	}

	if agent.lock == false {
		for j := range agent.weights {
			for _, value := range newStateActiveFeatures {
				agent.delta += agent.Gamma * epsilons[j] * agent.weights[j][value] // TD error target calculation
			}
		}

		var g float64
		var mhat float64
		var vhat float64

		// update for both actions for weights and traces
		for j := range agent.weights {
			for i := range agent.weights[j] {
				if agent.traces[j][i] != 0 { // update only where traces are non-zero
					if agent.IsStepsizeAdaptive == false {
						agent.weights[j][i] += agent.stepsize * agent.delta * agent.traces[j][i] // Semi-gradient descent, update weights
					} else {
						g = -agent.delta * agent.traces[j][i]
						agent.m[j][i] = agent.beta1*agent.m[j][i] + (1-agent.beta1)*g
						agent.v[j][i] = agent.beta1*agent.v[j][i] + (1-agent.beta1)*g*g

						mhat = agent.m[j][i] / (1 - math.Pow(agent.beta1, agent.timesteps))
						vhat = agent.v[j][i] / (1 - math.Pow(agent.beta2, agent.timesteps))
						agent.weights[j][i] -= agent.stepsize * mhat / (math.Pow(vhat, 0.5) + agent.e)
					}
					agent.traces[j][i] = agent.Gamma * agent.Lambda * agent.traces[j][i] // update traces
				}
			}
		}
	}

	// New information is old for the next time step
	agent.oldState = state
	agent.oldStateActiveFeatures = newStateActiveFeatures
	agent.oldAction = newAction

	if agent.EnableDebug {
		agent.Message("msg", "step", "state", state, "reward", reward, "action", agent.oldAction)
	}

	agent.timesteps++
	return agent.oldAction
}

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *ESarsa) End(state rlglue.State, reward float64) {
	agent.Step(state, reward)
	agent.traces = make([][]float64, 3) // one trace slice for each action
	agent.traces[0] = make([]float64, agent.tiler.NumberOfIndices())
	agent.traces[1] = make([]float64, agent.tiler.NumberOfIndices())
	agent.traces[2] = make([]float64, agent.tiler.NumberOfIndices())

	if agent.EnableDebug {
		agent.Message("msg", "end", "state", state, "reward", reward)
	}
}

// PolicyExpectedSarsaLambda returns action based on tile coded state
func (agent *ESarsa) PolicyExpectedSarsaLambda(tileCodedStateActiveFeatures []int) (rlglue.Action, []float64) {
	// Calculates action values
	actionValue0 := agent.ActionValue(tileCodedStateActiveFeatures, 0)
	actionValue1 := agent.ActionValue(tileCodedStateActiveFeatures, 1)

	greedyAction := 0
	if actionValue0 < actionValue1 {
		greedyAction = 1
	}

	// Calculates Epsilon-greedy probabilities for both actions
	probs := make([]float64, 2) // Probabilities of taking actions 0 and 1
	probs[(greedyAction+1)%2] = agent.Epsilon / 2
	probs[greedyAction] = 1 - probs[(greedyAction+1)%2]

	// Random sampling action based on epsilon-greedy policy
	var action rlglue.Action
	if agent.rng.Float64() >= probs[0] {
		action = 1
	} else {
		action = 0
	}

	return action, probs
}

// ActionValue returns action value for a tile coded state and action pair
func (agent *ESarsa) ActionValue(tileCodedStateActiveFeatures []int, action rlglue.Action) float64 {
	var actionValue float64

	// Calculates action value as linear function (dot product) between weights and binary featured state
	for _, value := range tileCodedStateActiveFeatures {
		a, _ := tpo.GetInt(action)
		actionValue += agent.weights[a][value]
	}

	return actionValue
}

func (agent *ESarsa) CheckAvgRwdLock(avg float64) bool {
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

func (agent *ESarsa) CheckAvgRwdUnlock(avg float64) bool {
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

func (agent *ESarsa) DynamicLock() bool {
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

func (agent *ESarsa) OnetimeRwdLock() bool {
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

func (agent *ESarsa) OnetimeEpLenLock() bool {
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

func (agent *ESarsa) KeepLock() bool {
	return true
}

func (agent *ESarsa) GetLock() bool {
	return agent.lock
}

func (agent *ESarsa) SaveWeights(basePath string) error {
	return nil
}
