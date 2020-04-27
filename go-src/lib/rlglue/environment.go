package rlglue

import "errors"

type Environment interface {
	// Initialize configures the environment with the provided parameters and resets any internal state.
	Initialize(Attributes, Logger) error

	// Start returns an initial observation.
	Start() State

	// Step takes an action and provides the new observation, the resulting reward, and whether the state is terminal.
	Step(Action) (State, float64, bool)

	// GetAttributes returns attributes for this environment.
	GetAttributes() Attributes
}

// NewEnvironmentCreator is a function that can create an environment.
type NewEnvironmentCreator func() (Environment, error)

// RegisterEnvironment is used to register a new environment type.
// This is most likely called by an init function in the Environment's go file.
// The function returns an error if an environment with that name already exists.
func RegisterEnvironment(name string, creator NewEnvironmentCreator) error {
	if _, ok := environmentList[name]; ok {
		return errors.New("Environment '" + name + "' has already been registered")
	}
	environmentList[name] = creator
	return nil
}

func CreateEnvironment(name string) (Environment, error) {
	creator, ok := environmentList[name]
	if !ok {
		return nil, errors.New("Environment '" + name + "' has not been registered")
	}
	return creator()
}

var environmentList = map[string]NewEnvironmentCreator{}
