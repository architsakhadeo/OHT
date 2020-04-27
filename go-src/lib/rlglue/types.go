package rlglue

import (
	"encoding/json"
	"io"
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
	LogStep(State, State, float64, ...float64)

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

// LoadSaver can be implemented by an Agent or Environment to save itself. If it's implemented, the
// Experiment will call these APIs when the config requests them.
type LoadSaver interface {
	// Save stores anything needed to save the struct (including relevant Config passed via Initialize)
	// into the provided Writer
	Save(io.Writer)

	// Load loads the struct from the provided Reader. Loading a saved struct must produce a struct
	// that behaves identically to the original struct, but with the new Logger.
	Load(io.Reader, Logger)
}

// FileLoadSaver can be implemented by an Agent or Environment to save itself. If it's implemented, the
// Experiment will call these APIs when the config requests them.
type FileLoadSaver interface {
	// Save stores anything needed to save the struct (including relevant Config passed via Initialize)
	// at the provided filepath.
	Save(string)

	// Load loads the struct from the provided filepath. Loading a saved struct must produce a struct
	// that behaves identically to the original struct, but with the new Logger.
	Load(string, Logger)
}
