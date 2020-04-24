package remote

import (
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

type Environment struct {
	logger rlglue.Logger
}

func NewEnvironment() rlglue.Environment {
	return &Environment{}
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (env *Environment) Initialize(attr rlglue.Attributes, logger rlglue.Logger) error {
	panic("environment.Initialize not implemented")
}

// Start returns an initial observation.
func (env *Environment) Start() rlglue.State {
	panic("environment.Start not implemented")
}

// Step takes an action and provides the resulting reward and new observation.
func (env *Environment) Step(action rlglue.Action) (float64, rlglue.State) {
	panic("environment.Step not implemented")
}

// GetAttributes returns attributes for this environment.
func (env *Environment) GetAttributes() rlglue.Attributes {
	panic("environment.GetAttributes not implemented")
}
