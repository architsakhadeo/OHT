package rlglue

import "io"

// Action is a single integer representing the action taken by the agent.
type Action int

// State is a slice of state observations.
type State []float64

// Logger can be used to log messages. TODO this should be a structured log.
type Logger interface {
	Message(string)
}

type Attributes map[string]interface{} // TODO this should probably be raw JSON which loads into structs

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
