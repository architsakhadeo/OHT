package rlglue

type Environment interface {
	// Initialize configures the environment with the provided parameters and resets any internal state.
	Initialize(Attributes, Logger) error

	// Start returns an initial observation.
	Start() State

	// Step takes an action and provides the resulting reward, the new observation, and whether the state is terminal.
	Step(Action) (float64, State, bool)

	// GetAttributes returns attributes for this environment.
	GetAttributes() Attributes
}
