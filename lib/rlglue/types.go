package rlglue

import (
	"encoding/json"
	"math"
)

// Action is a single integer representing the action taken by the agent.
type Action interface{}

// State is a slice of state observations.
type State []float64

type Attributes json.RawMessage

func (attr *Attributes) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, (*json.RawMessage)(attr))
}

const StatePrecision = 0.000001

func (s State) IsEqual(s2 State) bool {
	if s == nil {
		return s2 == nil
	}
	if s2 == nil || len(s) != len(s2) {
		return false
	}
	for i := range s {
		if math.Abs(s[i]-s2[i]) > StatePrecision {
			return false
		}
	}
	return true
}
