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

type Config map[string]interface{}

type LoadSaver interface{
	// Save stores anything needed to save the struct (including relevant Config passed via Initialize)
	// into the provided Writer
	Save(io.Writer)

	// Load loads the struct from the provided Reader. Loading a saved struct must produce a struct
	// that behaves identically to the original struct, but with the new Logger.
	Load(io.Reader, Logger)
}

type FileLoadSaver interface{
	// Save stores anything needed to save the struct (including relevant Config passed via Initialize)
	// at the provided filepath.
	Save(string)

	// Load loads the struct from the provided filepath. Loading a saved struct must produce a struct
	// that behaves identically to the original struct, but with the new Logger.
	Load(string, Logger)
}
