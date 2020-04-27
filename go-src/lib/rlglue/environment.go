package rlglue

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
