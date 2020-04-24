package example

import (
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
func (env *Environment) Initialize(config rlglue.Config, logger rlglue.Logger) {
	env.logger = logger

	var seed int64
	if sd, ok := config["seed"]; !ok {
		// Config doesn't have a seed
		seed = 0
	} else if seed, ok = sd.(int64); !ok {
		// Config seed is wrong type
		logger.Message("example.Environment seed was of wrong type")
		seed = 0
	}
	rand.Seed(seed)
	env.state = rand.Intn(NumberOfActions) - ActionMax
}

// Start returns an initial observation.
func (env *Environment) Start() rlglue.State {
	return env.stateSlice()
}

// Step takes an action and provides the resulting reward and new observation.
func (env *Environment) Step(act rlglue.Action) (float64, rlglue.State) {
	action := int(act)
	if action < -ActionMax || action > ActionMax {
		return 0, rlglue.State{} //, error.New("example.Environment action must be between -10 and 10")
	}
	env.state += action - 10
	if env.state >= StateMax || env.state <= -StateMax {
		env.state = 0
	}

	return float64(env.state), env.stateSlice()
}

// GetAttributes returns attributes for this environment.
func (env *Environment) GetAttributes() rlglue.EnvironmentAttributes {
	return rlglue.EnvironmentAttributes{
		NumberOfActions:  NumberOfActions,
		DimensionOfState: 1,
		StateRange:       []rlglue.State{[]float64{float64(StateMax) * 2}},
	}
}

func (env *Environment) stateSlice() rlglue.State {
	return rlglue.State([]float64{float64(env.state)})
}
