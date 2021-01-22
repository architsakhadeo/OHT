package rlglue

type Agent interface {
	// Initialize configures the agent with the provided parameters and resets any internal state.
	// The run number is used in case many agents are run simultaneously, e.g. to modify the random seed.
	// The first attributes are experimental attributes; the second are environmental.
	Initialize(run uint, experiment Attributes, environment Attributes) error

	// Start provides an initial observation to the agent and returns the agent's action.
	Start(state State) Action

	// Step provides a new observation and a reward to the agent and returns the agent's next action.
	Step(state State, reward float64) Action

	// End informs the agent that a terminal state has been reached, providing the final reward.
	End(state State, reward float64)

	GetLock() bool

	// Save function approximator weights at the end of learning phase.
	SaveWeights(basePath string) error
}
