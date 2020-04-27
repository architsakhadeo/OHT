package example

import (
	"encoding/json"
	"math/rand"

	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

const (
	ActionMax       = 10
	StateMax        = ActionMax * 10
	NumberOfActions = 2*ActionMax + 1
)

// Environment is just a dumb test environment.
// state is adjusted by the action. Valid values are integers [0, NumberOfActions).
// Reward is equal to the new state.
// When |state| is >= StateMax, it's reset to 0.
type Environment struct {
	logger rlglue.Logger
	state  int
}

func NewEnvironment() rlglue.Environment {
	return &Environment{}
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (env *Environment) Initialize(attr rlglue.Attributes, logger rlglue.Logger) error {
	env.logger = logger

	var seed int64
	err := json.Unmarshal(attr, &seed)
	if err != nil {
		logger.Message("example.Agent seed wasn't available")
		seed = 0
	}
	rand.Seed(seed)

	env.state = rand.Intn(NumberOfActions) - ActionMax

	return nil
}

// Start returns an initial observation.
func (env *Environment) Start() rlglue.State {
	return env.stateSlice()
}

// Step takes an action and provides the resulting reward, the new observation, and whether the state is terminal.
// For this continuous environment, it's only terminal if the action was invalid.
func (env *Environment) Step(act rlglue.Action) (rlglue.State, float64, bool) {
	action := int(act)
	if action < -ActionMax || action > ActionMax {
		return rlglue.State{}, 0, true //, error.New("example.Environment action must be between -10 and 10")
	}
	env.state += action - 10
	if env.state >= StateMax || env.state <= -StateMax {
		env.state = 0
	}

	return env.stateSlice(), float64(env.state), false
}

// GetAttributes returns attributes for this environment.
func (env *Environment) GetAttributes() rlglue.Attributes {
	return rlglue.Attributes{
		// TODO should be saved as attributes from a known struct
		// NumberOfActions:  NumberOfActions,
		// DimensionOfState: 1,
		// StateRange:       []rlglue.State{[]float64{float64(StateMax) * 2}},
	}
}

func (env *Environment) stateSlice() rlglue.State {
	return rlglue.State([]float64{float64(env.state)})
}
