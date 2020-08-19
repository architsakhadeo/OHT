package rlglue

import "io"

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
}

// PersistentAgent is an agent that can be persisted to a file. The data format is not specified.
type PersistentAgent interface {
	Agent

	// Save saves the agent to the provided writer.
	Save(wr io.Writer) error

	// Load loads the agent from the provided reader. If used, it should only be called before
	// the Agent's Initialize is called. If the experiment or environment attributes are persisted,
	// Initialize might need to check if the loaded attributes match the ones provided to
	// Initialize, or it might handle conflicts in some way.
	Load(rd io.Reader) error
}
