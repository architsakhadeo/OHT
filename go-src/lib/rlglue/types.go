package rlglue

import (
	"encoding/json"
)

// Action is a single integer representing the action taken by the agent.
type Action int

// State is a slice of state observations.
type State []float64

// Logger can be used to log messages.
type Logger interface {
	// Message logs a message followed by pairs of string-value to be stored in a structured log.
	Message(string, ...interface{})

	// MessageDelta calls Message and appends the time since the last Message or MessageDelta.
	MessageDelta(string, ...interface{})

	// MessageRewardSince calls Message and appends the reward since the provided step (calculated from the
	// reward log).
	MessageRewardSince(int, ...interface{})

	// LogEpisodeLength adds the provided episode length to the episode length log.
	LogEpisodeLength(int)

	// LogStepHeader lists the headers used in the optional variadic arguments to LogStep.
	LogStepHeader(...string)

	// LogStep adds information from a step to the step log. It must contain previous state, current state,
	// and reward. It can optionally add other float64 values to be logged. (If so, LogStepHeader must be
	// called to provide headers and so the logger knows how many to expect.)
	LogStep(State, State, Action, float64, ...float64)

	// Interval gives the desired number of steps to take between logging messages.
	// This number is constant, so it should be cached for efficiency.
	Interval() int

	// Save persists the logged information to disk.
	Save()
}

type Attributes json.RawMessage

func (attr *Attributes) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, (*json.RawMessage)(attr))
}
