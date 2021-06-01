package agent

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"

	tpo "github.com/stellentus/cartpoles/lib/util/type-opr"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	"github.com/stellentus/cartpoles/lib/util"
)

const (
	maxFeatureAcrobot1 = 1.0
	maxFeatureAcrobot2 = 1.0
	maxFeatureAcrobot3 = 1.0
	maxFeatureAcrobot4 = 1.0
	maxFeatureAcrobot5 = 4.0 * math.Pi
	maxFeatureAcrobot6 = 9.0 * math.Pi
)

type esarsaAcrobotSettings struct {
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
}

// Expected sarsa-lambda with tile coding
type ESarsaAcrobot struct {
	logger.Debug
	rng   *rand.Rand
	tiler util.MultiTiler

	// Agent accessible parameters
	weights                [][]float64 // weights is a slice of weights for each action
	traces                 [][]float64
	delta                  float64
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
	esarsaAcrobotSettings
}

func init() {
	Add("esarsa_acrobot", NewESarsaAcrobot)
}

func NewESarsaAcrobot(logger logger.Debug) (rlglue.Agent, error) {
	return &ESarsaAcrobot{Debug: logger}, nil
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *ESarsaAcrobot) Initialize(run uint, expAttr, envAttr rlglue.Attributes) error {
	agent.esarsaAcrobotSettings = esarsaAcrobotSettings{
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

	err := json.Unmarshal(expAttr, &agent.esarsaAcrobotSettings)
	if err != nil {
		agent.Message("warning", "agent.ESarsa settings weren't available: "+err.Error())
		agent.esarsaAcrobotSettings.Seed = 0
	}

	if agent.IsStepsizeAdaptive == false {
		agent.stepsize = agent.Alpha / float64(agent.esarsaAcrobotSettings.NumTilings) // Setting stepsize
	} else {
		agent.stepsize = agent.AdaptiveAlpha / float64(agent.esarsaAcrobotSettings.NumTilings) // Setting adaptive stepsize
	}

	agent.beta1 = 0.9
	agent.beta2 = 0.999
	agent.e = math.Pow(10, -8)

	agent.esarsaAcrobotSettings.Seed += int64(run)
	agent.rng = rand.New(rand.NewSource(agent.esarsaAcrobotSettings.Seed)) // Create a new rand source for reproducibility

	// scales the input observations for tile-coding
	scalers := []util.Scaler{
		util.NewScaler(-maxFeatureAcrobot1, maxFeatureAcrobot1, agent.esarsaAcrobotSettings.NumTiles),
		util.NewScaler(-maxFeatureAcrobot2, maxFeatureAcrobot2, agent.esarsaAcrobotSettings.NumTiles),
		util.NewScaler(-maxFeatureAcrobot3, maxFeatureAcrobot3, agent.esarsaAcrobotSettings.NumTiles),
		util.NewScaler(-maxFeatureAcrobot4, maxFeatureAcrobot4, agent.esarsaAcrobotSettings.NumTiles),
		util.NewScaler(-maxFeatureAcrobot5, maxFeatureAcrobot5, agent.esarsaAcrobotSettings.NumTiles),
		util.NewScaler(-maxFeatureAcrobot6, maxFeatureAcrobot6, agent.esarsaAcrobotSettings.NumTiles),
	}

	agent.tiler, err = util.NewMultiTiler(6, agent.esarsaAcrobotSettings.NumTilings, scalers)
	if err != nil {
		return err
	}
	agent.weights = make([][]float64, 3) // one weight slice for each action
	agent.weights[0] = make([]float64, agent.tiler.NumberOfIndices())
	agent.weights[1] = make([]float64, agent.tiler.NumberOfIndices())
	agent.weights[2] = make([]float64, agent.tiler.NumberOfIndices())

	agent.traces = make([][]float64, 3) // one trace slice for each action
	agent.traces[0] = make([]float64, agent.tiler.NumberOfIndices())
	agent.traces[1] = make([]float64, agent.tiler.NumberOfIndices())
	agent.traces[2] = make([]float64, agent.tiler.NumberOfIndices())

	agent.m = make([][]float64, 3)
	agent.m[0] = make([]float64, agent.tiler.NumberOfIndices())
	agent.m[1] = make([]float64, agent.tiler.NumberOfIndices())
	agent.m[2] = make([]float64, agent.tiler.NumberOfIndices())

	agent.v = make([][]float64, 3)
	agent.v[0] = make([]float64, agent.tiler.NumberOfIndices())
	agent.v[1] = make([]float64, agent.tiler.NumberOfIndices())
	agent.v[2] = make([]float64, agent.tiler.NumberOfIndices())

	agent.timesteps = 0

	agent.Message("esarsa acrobot settings", fmt.Sprintf("%+v", agent.esarsaAcrobotSettings))

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *ESarsaAcrobot) Start(state rlglue.State) rlglue.Action {

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
func (agent *ESarsaAcrobot) Step(state rlglue.State, reward float64) rlglue.Action {
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

	// New information is old for the next time step
	agent.oldStateActiveFeatures = newStateActiveFeatures
	agent.oldAction = newAction

	if agent.EnableDebug {
		agent.Message("msg", "step", "state", state, "reward", reward, "action", agent.oldAction)
	}

	agent.timesteps++
	return agent.oldAction
}

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *ESarsaAcrobot) End(state rlglue.State, reward float64) {
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
func (agent *ESarsaAcrobot) PolicyExpectedSarsaLambda(tileCodedStateActiveFeatures []int) (rlglue.Action, []float64) {
	// Calculates action values
	actionValue0 := agent.ActionValue(tileCodedStateActiveFeatures, 0)
	actionValue1 := agent.ActionValue(tileCodedStateActiveFeatures, 1)
	actionValue2 := agent.ActionValue(tileCodedStateActiveFeatures, 2)

	greedyAction := agent.findArgmax([]float64{actionValue0, actionValue1, actionValue2})

	// Calculates Epsilon-greedy probabilities for both actions
	probs := make([]float64, 3) // Probabilities of taking actions 0 and 1
	for i := range probs {
		probs[i] = agent.Epsilon / 3
	}
	probs[greedyAction] = 1 - agent.Epsilon + agent.Epsilon/3

	// Random sampling action based on epsilon-greedy policy
	var action rlglue.Action
	var randomval float64
	randomval = agent.rng.Float64()
	if randomval <= probs[0] {
		action = 0
	} else if randomval > probs[0] && randomval <= probs[0]+probs[1] {
		action = 1
	} else {
		action = 2
	}

	return action, probs
}

// ActionValue returns action value for a tile coded state and action pair
func (agent *ESarsaAcrobot) ActionValue(tileCodedStateActiveFeatures []int, action rlglue.Action) float64 {
	var actionValue float64

	// Calculates action value as linear function (dot product) between weights and binary featured state
	for _, value := range tileCodedStateActiveFeatures {
		a, _ := tpo.GetInt(action)
		actionValue += agent.weights[a][value]
	}

	return actionValue
}

func (agent *ESarsaAcrobot) GetLock() bool {
	return false
}

func (agent *ESarsaAcrobot) findArgmax(array []float64) int {
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

func (agent *ESarsaAcrobot) SaveWeights(basePath string) error {
	return nil
}

func (agent *ESarsaAcrobot) GetLearnProg() string {
	return "0"
}

func (agent *ESarsaAcrobot) PassInfo(info string, value float64) interface{} {
	return nil
}
