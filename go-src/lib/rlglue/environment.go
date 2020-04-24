package rlglue

type Environment interface {
	// Initialize configures the environment with the provided parameters and resets any internal state.
	Initialize(Config, Logger)

	// Start returns an initial observation.
	Start() State

	// Step takes an action and provides the resulting reward and new observation.
	Step(Action) (float64, State)

	// GetAttributes returns attributes for this environment.
	GetAttributes() EnvironmentAttributes
}

type EnvironmentAttributes struct {
	// NumberOfActions is the number of discrete actions that can be taken in the environment.
	NumberOfActions int

	// DimensionOfState is the number of elements in the State slice.
	DimensionOfState int

	// StateRange is a State slice of the range of valid state values.
	// The specification for this should be improved. A range is usually a minimum and maximum.
	StateRange []State

	// Custom can be used for any non-standard environment attributes.
	Custom map[string]interface{}
}
