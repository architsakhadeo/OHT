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
	// Cartpole constants
	maxPosition        = 2.4
	maxVelocity        = 4
	maxAngle           = 12 * 2 * math.Pi / 360
	maxAngularVelocity = 3.5

	// Acrobot constants
	maxFeature1 = 1.0
	maxFeature2 = 1.0
	maxFeature3 = 1.0
	maxFeature4 = 1.0
	maxFeature5 = 4.0 * math.Pi
	maxFeature6 = 9.0 * math.Pi
)

type EsarsaSettings struct {
	EnableDebug        bool    `json:"enable-debug"`
	Seed               int64   `json:"seed"`
	TotalRuns    	   uint    `json:"total-runs"`
	NumTilings         int     `json:"tilings"`
	NumTiles           int     `json:"tiles"`
	Gamma              float64 `json:"gamma"`
	Lambda             float64 `json:"lambda"`
	Epsilon            float64 `json:"epsilon"`
	EpsilonAfterLock   float64 `json:"epsilon-after-lock"`
	Alpha              float64 `json:"alpha"`
	AdaptiveAlpha      float64 `json:"adaptive-alpha"`
	IsStepsizeAdaptive bool    `json:"is-stepsize-adaptive"`
	EnvName            string  `json:"env-name"`

	StateDim int    `json:"state-len"`
	Bsize    int    `json:"buffer-size"`
	Btype    string `json:"buffer-type"`

	NumActions int     `json:"numberOfActions"`
	WInit      float64 `json:"weight-init"`
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
	EsarsaSettings

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
	} else if lw.LockCondition == "onetime-epstep-lessthan" {
		lw.CheckChange = agent.OnetimeEpStepLessThanLock
	} else if lw.LockCondition == "onetime-epstep-greaterthan" {
		lw.CheckChange = agent.OnetimeEpStepGreaterThanLock
	}
	return lw
}

func DefaultESarsaSettings() EsarsaSettings {
	return EsarsaSettings{
		// These default settings will be used if the config doesn't set these values
		NumTilings:         32,
		NumTiles:           4,
		NumActions:         2,
		Gamma:              0.99,
		Lambda:             0.8,
		Epsilon:            0.05,
		EpsilonAfterLock:   0.05,
		Alpha:              0.01,
		AdaptiveAlpha:      0.001,
		IsStepsizeAdaptive: false,
		WInit:              0.0,
		EnvName:            "cartpole",
	}
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *ESarsa) Initialize(run uint, expAttr, envAttr rlglue.Attributes) error {
	set := DefaultESarsaSettings()
	err := json.Unmarshal(expAttr, &set)
	if err != nil {
		agent.Message("warning", "agent.ESarsa settings weren't available: "+err.Error())
		set.Seed = 0
	}
	if set.TotalRuns != 0 {
		set.Seed += int64(run / set.TotalRuns)
	} else {
		set.Seed += int64(run)
	}

	lw := lockweight.LockWeight{}
	err = json.Unmarshal(expAttr, &lw)
	if err != nil {
		return errors.New("ESarsa agent LockWeight attributes were not valid: " + err.Error())
	}

	return agent.InitializeWithSettings(set, lw)
}

func (agent *ESarsa) InitializeWithSettings(set EsarsaSettings, lw lockweight.LockWeight) error {
	agent.EsarsaSettings = set
	agent.lw = lw

	agent.bf = buffer.NewBuffer()
	agent.bf.Initialize(agent.Btype, agent.Bsize, agent.StateDim, agent.Seed)
	agent.lw = agent.InitLockWeight(agent.lw)

	if agent.EsarsaSettings.IsStepsizeAdaptive == false {
		agent.stepsize = agent.EsarsaSettings.Alpha / float64(agent.EsarsaSettings.NumTilings) // Setting stepsize
	} else {
		agent.stepsize = agent.EsarsaSettings.AdaptiveAlpha / float64(agent.EsarsaSettings.NumTilings) // Setting adaptive stepsize
	}

	agent.beta1 = 0.9
	agent.beta2 = 0.999
	agent.e = math.Pow(10, -8)

	//fmt.Println("Agent Seed: ", agent.Seed)
	//fmt.Println("Agent EsarsaSettings Seed: ", agent.EsarsaSettings.Seed)
	//fmt.Println("Set Seed:", set.Seed)
	//fmt.Println("Seed actually used by the agent:", agent.EsarsaSettings.Seed)
	agent.rng = rand.New(rand.NewSource(agent.EsarsaSettings.Seed)) // Create a new rand source for reproducibility

	// scales the input observations for tile-coding
	var err error
	if agent.EsarsaSettings.EnvName == "cartpole" {
		agent.NumActions = 2
		scalers := []util.Scaler{
			util.NewScaler(-maxPosition, maxPosition, agent.EsarsaSettings.NumTiles),
			util.NewScaler(-maxVelocity, maxVelocity, agent.EsarsaSettings.NumTiles),
			util.NewScaler(-maxAngle, maxAngle, agent.EsarsaSettings.NumTiles),
			util.NewScaler(-maxAngularVelocity, maxAngularVelocity, agent.EsarsaSettings.NumTiles),
		}

		agent.tiler, err = util.NewMultiTiler(4, agent.EsarsaSettings.NumTilings, scalers)
		if err != nil {
			return err
		}
	} else if agent.EsarsaSettings.EnvName == "acrobot" {
		agent.NumActions = 2 //3
		scalers := []util.Scaler{
			util.NewScaler(-maxFeature1, maxFeature1, agent.EsarsaSettings.NumTiles),
			util.NewScaler(-maxFeature2, maxFeature2, agent.EsarsaSettings.NumTiles),
			util.NewScaler(-maxFeature3, maxFeature3, agent.EsarsaSettings.NumTiles),
			util.NewScaler(-maxFeature4, maxFeature4, agent.EsarsaSettings.NumTiles),
			util.NewScaler(-maxFeature5, maxFeature5, agent.EsarsaSettings.NumTiles),
			util.NewScaler(-maxFeature6, maxFeature6, agent.EsarsaSettings.NumTiles),
		}

		agent.tiler, err = util.NewMultiTiler(6, agent.EsarsaSettings.NumTilings, scalers)
		if err != nil {
			return err
		}
	}

	agent.weights = make([][]float64, agent.NumActions) // one weight slice for each action
	for i := 0; i < agent.NumActions; i++ {
		agent.weights[i] = make([]float64, agent.tiler.NumberOfIndices())
	}

	agent.traces = make([][]float64, agent.NumActions) // one trace slice for each action
	for i := 0; i < agent.NumActions; i++ {
		agent.traces[i] = make([]float64, agent.tiler.NumberOfIndices())
	}

	agent.m = make([][]float64, agent.NumActions)
	for i := 0; i < agent.NumActions; i++ {
		agent.m[i] = make([]float64, agent.tiler.NumberOfIndices())
	}

	agent.v = make([][]float64, agent.NumActions)
	for i := 0; i < agent.NumActions; i++ {
		agent.v[i] = make([]float64, agent.tiler.NumberOfIndices())
	}

	agent.timesteps = 0

	for i := 0; i < len(agent.weights); i++ {
		for j := 0; j < len(agent.weights[0]); j++ {
			agent.weights[i][j] = agent.EsarsaSettings.WInit / float64(agent.EsarsaSettings.NumTilings)
		}
	}
	agent.Message("esarsa settings", fmt.Sprintf("%+v", agent.EsarsaSettings))

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
		agent.bf.Feed(agent.oldState, agent.oldAction, state, reward, agent.EsarsaSettings.Gamma)
		agent.lock = agent.lw.CheckChange()
		if agent.lock == true {
			agent.Epsilon = agent.EpsilonAfterLock
		}
	}

	if agent.lock == false {
		for j := range agent.weights {
			for _, value := range newStateActiveFeatures {
				agent.delta += agent.EsarsaSettings.Gamma * epsilons[j] * agent.weights[j][value] // TD error target calculation
			}
		}

		var g float64
		var mhat float64
		var vhat float64

		// update for both actions for weights and traces
		for j := range agent.weights {
			for i := range agent.weights[j] {
				if agent.traces[j][i] != 0 { // update only where traces are non-zero
					if agent.EsarsaSettings.IsStepsizeAdaptive == false {
						agent.weights[j][i] += agent.stepsize * agent.delta * agent.traces[j][i] // Semi-gradient descent, update weights
					} else {
						g = -agent.delta * agent.traces[j][i]
						agent.m[j][i] = agent.beta1*agent.m[j][i] + (1-agent.beta1)*g
						agent.v[j][i] = agent.beta1*agent.v[j][i] + (1-agent.beta1)*g*g

						mhat = agent.m[j][i] / (1 - math.Pow(agent.beta1, agent.timesteps))
						vhat = agent.v[j][i] / (1 - math.Pow(agent.beta2, agent.timesteps))
						agent.weights[j][i] -= agent.stepsize * mhat / (math.Pow(vhat, 0.5) + agent.e)
					}
					agent.traces[j][i] = agent.EsarsaSettings.Gamma * agent.EsarsaSettings.Lambda * agent.traces[j][i] // update traces
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
		agent.bf.Feed(agent.oldState, agent.oldAction, state, reward, 0.0)
		agent.lock = agent.lw.CheckChange()
		if agent.lock == true {
			agent.Epsilon = agent.EpsilonAfterLock
		}
	}

	if agent.lock == false {
		for j := range agent.weights {
			for _, value := range newStateActiveFeatures {
				agent.delta += 0.0 * epsilons[j] * agent.weights[j][value] // TD error target calculation
			}
		}

		var g float64
		var mhat float64
		var vhat float64

		// update for both actions for weights and traces
		for j := range agent.weights {
			for i := range agent.weights[j] {
				if agent.traces[j][i] != 0 { // update only where traces are non-zero
					if agent.EsarsaSettings.IsStepsizeAdaptive == false {
						agent.weights[j][i] += agent.stepsize * agent.delta * agent.traces[j][i] // Semi-gradient descent, update weights
					} else {
						g = -agent.delta * agent.traces[j][i]
						agent.m[j][i] = agent.beta1*agent.m[j][i] + (1-agent.beta1)*g
						agent.v[j][i] = agent.beta1*agent.v[j][i] + (1-agent.beta1)*g*g

						mhat = agent.m[j][i] / (1 - math.Pow(agent.beta1, agent.timesteps))
						vhat = agent.v[j][i] / (1 - math.Pow(agent.beta2, agent.timesteps))
						agent.weights[j][i] -= agent.stepsize * mhat / (math.Pow(vhat, 0.5) + agent.e)
					}
					agent.traces[j][i] = 0.0 * agent.EsarsaSettings.Lambda * agent.traces[j][i] // update traces
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

	agent.traces = make([][]float64, agent.NumActions) // one trace slice for each action
	for i := 0; i < agent.NumActions; i++ {
		agent.traces[i] = make([]float64, agent.tiler.NumberOfIndices())
	}

	if agent.EnableDebug {
		agent.Message("msg", "end", "state", state, "reward", reward)
	}
}

// PolicyExpectedSarsaLambda returns action based on tile coded state
func (agent *ESarsa) PolicyExpectedSarsaLambda(tileCodedStateActiveFeatures []int) (rlglue.Action, []float64) {
	// Calculates action values
	actionValues := make([]float64, agent.NumActions)
	for i := 0; i < agent.NumActions; i++ {
		actionValues[i] = agent.ActionValue(tileCodedStateActiveFeatures, i)
	}
	//fmt.Println("action value", actionValue0, actionValue1)

	greedyAction := agent.findArgmax(actionValues)
	//if actionValues[0] < actionValues[1] {
	//	greedyAction = 1
	//} else if actionValues[0] == actionValues[1] {
	//	greedyAction = agent.rng.Int() % 2 //agent.EsarsaSettings.NumActions
	//}
	probs := make([]float64, agent.NumActions) // Probabilities of taking actions 0 and 1
	for i := range probs {
		probs[i] = (agent.Epsilon / float64(agent.NumActions))
	}
	probs[greedyAction] = 1 - agent.Epsilon + (agent.Epsilon / float64(agent.NumActions))

	cummulativeProbs := make([]float64, agent.NumActions)
	sum := 0.0
	for i := range probs {
		sum += probs[i]
		cummulativeProbs[i] = sum
	}
	// Calculates Epsilon-greedy probabilities for both actions
	//probs := make([]float64, agent.NumActions) // Probabilities of taking actions 0 and 1
	//probs[(greedyAction+1)%2] = agent.Epsilon / 2
	//probs[greedyAction] = 1 - probs[(greedyAction+1)%2]

	// Random sampling action based on epsilon-greedy policy
	var action rlglue.Action
	flag := false
	randomval := agent.rng.Float64()

	for i := 0; i < agent.NumActions-1; i++ {
		if randomval <= cummulativeProbs[i] {
			action = i
			flag = true
			break
		}
	}
	if flag == false {
		action = agent.NumActions - 1
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

func (agent *ESarsa) OnetimeEpStepLessThanLock() bool {
	if agent.lock {
		return true
	} else {
		_, _, _, _, gammas2D := agent.bf.Content()
		gammas := ao.Flatten2DFloat(gammas2D)
		count := agent.Count(gammas, 0.0)
		avgEpSteps := float64(len(gammas)) / float64(count)
		if len(gammas) < agent.Bsize {
			return false
		}
		if avgEpSteps <= agent.lw.LockAvgEpStepLessThan {
			return true
		}
		return false
	}
}

func (agent *ESarsa) OnetimeEpStepGreaterThanLock() bool {
	if agent.lock {
		return true
	} else {
		_, _, _, _, gammas2D := agent.bf.Content()
		gammas := ao.Flatten2DFloat(gammas2D)
		count := agent.Count(gammas, 0.0)
		avgEpSteps := float64(len(gammas)) / float64(count)
		if len(gammas) < agent.Bsize {
			return false
		}
		if avgEpSteps >= agent.lw.LockAvgEpStepGreaterThan {
			return true
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

func (agent *ESarsa) findArgmax(array []float64) int {
	max := array[0]
	argmax := 0
	for i, value := range array {
		if value > max {
			max = value
			argmax = i
		}
	}
	return argmax
}

func (agent *ESarsa) Count(array []float64, number float64) int {
	var count int
	count = 0
	for _, value := range array {
		if value == number {
			count++
		}
	}
	return count
}

func (agent *ESarsa) SaveWeights(basePath string) error {
	return nil
}

func (agent *ESarsa) GetLearnProg() float64 {
	return float64(0)
}
