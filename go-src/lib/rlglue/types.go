package rlglue

import (
	"encoding/json"
)

// Action is a single integer representing the action taken by the agent.
type Action int

// State is a slice of state observations.
type State []float64

type Attributes json.RawMessage

func (attr *Attributes) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, (*json.RawMessage)(attr))
}
