package environment

import (
	"encoding/json"
	"math"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

type futileSettings struct {
	Seed int64  `json:"seed"`
	Name string `json:"name"`
}

// Futile is a futile environment for learning offline. Both state and reward values are not used by the agent.
// Use Futile instead of Replay env if the agent is FQI, where batches of data are sampled from the dataset.
type Futile struct {
	logger.Debug
	futileSettings
	state []float64
}

func init() {
	Add("futile", NewFutile)
}

func NewFutile(logger logger.Debug) (rlglue.Environment, error) {
	return &Futile{Debug: logger}, nil
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (env *Futile) Initialize(run uint, attr rlglue.Attributes) error {
	err := json.Unmarshal(attr, &env.futileSettings)
	if err != nil {
		env.Message("warning", "environment.Futile seed wasn't available")
		env.Seed = 0
	}
	switch env.Name {
	case "cartpole":
		env.state = make(rlglue.State, 4)
	case "acrobot":
		env.state = make(rlglue.State, 6)
	}

	return nil
}

// Start returns an initial observation.
func (env *Futile) Start(randomizeStartStateCondition bool) rlglue.State {
	return env.state
}

// Step takes an action and provides the resulting reward, the new observation, and whether the state is terminal.
func (env *Futile) Step(act rlglue.Action, randomizeStartStateCondition bool) (rlglue.State, float64, bool) {
	return env.state, 0, false
}

// GetAttributes returns attributes for this environment.
func (env *Futile) GetAttributes() rlglue.Attributes {
	var attributes struct {
		NumAction  int       `json:"numberOfActions"`
		StateDim   int       `json:"stateDimension"`
		StateRange []float64 `json:"stateRange"`
	}
	switch env.Name {
	case "cartpole":
		attributes.NumAction = 2
		attributes.StateDim = 4
		attributes.StateRange = []float64{4.8, 8.0, (2 * 12 * 2 * math.Pi / 360), 7.0}
	case "acrobot":
		attributes.NumAction = 3
		attributes.StateDim = 6
		attributes.StateRange = []float64{2.0, 2.0, 2.0, 2.0, 2.0 * maxVel1, 2.0 * maxVel2}
	}

	attr, err := json.Marshal(&attributes)
	if err != nil {
		env.Message("err", "environment.Futile could not Marshal its JSON attributes: "+err.Error())
	}

	return attr
}
