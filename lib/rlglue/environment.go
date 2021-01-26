package rlglue

type Environment interface {
	// Initialize configures the environment with the provided parameters and resets any internal state.
	// The run number is used in case many agents are run simultaneously, e.g. to modify the random seed.
	Initialize(run uint, attr Attributes) error

	// Start returns an initial observation.
	Start(randomizeStartStateCondition bool) State

	// Step takes an action and provides the new observation, the resulting reward, and whether the state is terminal.
	Step(action Action, randomizeStartStateCondition bool) (State, float64, bool)

	// GetAttributes returns attributes for this environment.
	GetAttributes() Attributes
}
