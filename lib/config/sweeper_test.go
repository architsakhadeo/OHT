package config

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	name  = json.RawMessage("\"test\"")
	gamma = json.RawMessage("0.9")
	alpha = json.RawMessage("15")
	sweep = json.RawMessage(`{
			"alpha": [1e-3, 1e-4, 1e-5],
			"fqi-hidden": [
				[4, 4], [16, 16]
			],
			"epsilon": [0]
		}`)
)

func newTestAttributeMap() AttributeMap {
	return AttributeMap{
		"name":  &name,
		"gamma": &gamma,
		"alpha": &alpha,
		"sweep": &sweep,
	}
}

func TestAttributeMapCopy(t *testing.T) {
	am := newTestAttributeMap()
	assert.Equal(t, am, am.Copy())
}

func ExampleAttributeMap_String() {
	am := newTestAttributeMap()
	fmt.Println(am.String())
	// Output:
	// AM<alpha:15, gamma:0.9, name:"test", sweep:{
	// 			"alpha": [1e-3, 1e-4, 1e-5],
	// 			"fqi-hidden": [
	// 				[4, 4], [16, 16]
	// 			],
	// 			"epsilon": [0]
	// 		}>
}
