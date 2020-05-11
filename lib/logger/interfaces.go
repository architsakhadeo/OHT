package logger

import "github.com/stellentus/cartpoles/lib/rlglue"

// Data can be used to log data.
type Data interface {

	// LogEpisodeLength adds the provided episode length to the episode length log.
	LogEpisodeLength(int)

	// LogStep adds information from a step to the step log. It must contain previous state, current state,
	// and reward.
	LogStep(prevState, newState rlglue.State, action rlglue.Action, reward float64)

	// LogStepHeader lists the headers used in the optional variadic arguments to LogStepMulti.
	LogStepHeader(...string)

	// LogStepMulti is like LogStep, but it can optionally add other float64 values to be logged. (If so,
	// LogStepHeader must be called to provide headers and so the logger knows how many to expect.)
	LogStepMulti(prevState, newState rlglue.State, action rlglue.Action, reward float64, others ...float64)

	// RewardSince returns the total reward since the provided step (calculated from the reward log).
	RewardSince(int) float64

	// SaveLog persists the logged information to disk.
	SaveLog() error
}

// Debug can be used to log debug.
type Debug interface {
	// Message logs pairs of string-value to be stored in a structured log.
	Message(...interface{})

	// MessageDelta calls Message and appends the time since the last Message or MessageDelta.
	MessageDelta(...interface{})

	// Error logs an error if not nil.
	Error(err *error)
}
