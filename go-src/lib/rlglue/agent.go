package rlglue

type Agent interface {
	// Initialize configures the agent with the provided parameters and resets any internal state.
	Initialize(Config, EnvironmentAttributes, Logger)

	// Start provides an initial observation to the agent and returns the agent's action.
	Start(state State) Action

	// Step provides a new observation and a reward to the agent and returns the agent's next action.
	Step(reward float64, state State) Action

	// End informs the agent that a terminal state has been reached, providing the final reward.
	End(reward float64)
}
