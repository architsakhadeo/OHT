package config

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stellentus/cartpoles/lib/rlglue"
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
	epsilon = json.RawMessage("0")
	alpha3  = json.RawMessage("1e-3")
	alpha4  = json.RawMessage("1e-4")
	alpha5  = json.RawMessage("1e-5")
	fqi4    = json.RawMessage("[4, 4]")
	fqi16   = json.RawMessage("[16, 16]")
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

func newTestRlglueAttributes() rlglue.Attributes {
	return rlglue.Attributes(`{
		"name": "test",
		"gamma": 0.9,
		"alpha": 15,
		"sweep": {
			"alpha": [1e-3, 1e-4, 1e-5],
			"fqi-hidden": [
				[4, 4], [16, 16]
			],
			"epsilon": [0]
		}
	}`)
}

func TestAttrAsJson(t *testing.T) {
	raw := newTestRlglueAttributes()
	out := AttributeMap{}
	err := json.Unmarshal(raw, &out)
	assert.NoError(t, err)
	assert.Equal(t, newTestAttributeMap(), out)
}

func TestSweeperLoad(t *testing.T) {
	sw := sweeper{}
	err := sw.Load(newTestRlglueAttributes())
	assert.NoError(t, err)

	expected := []AttributeMap{
		AttributeMap{"name": &name, "gamma": &gamma, "alpha": &alpha3, "fqi-hidden": &fqi4, "epsilon": &epsilon},
		AttributeMap{"name": &name, "gamma": &gamma, "alpha": &alpha3, "fqi-hidden": &fqi16, "epsilon": &epsilon},
		AttributeMap{"name": &name, "gamma": &gamma, "alpha": &alpha4, "fqi-hidden": &fqi4, "epsilon": &epsilon},
		AttributeMap{"name": &name, "gamma": &gamma, "alpha": &alpha4, "fqi-hidden": &fqi16, "epsilon": &epsilon},
		AttributeMap{"name": &name, "gamma": &gamma, "alpha": &alpha5, "fqi-hidden": &fqi4, "epsilon": &epsilon},
		AttributeMap{"name": &name, "gamma": &gamma, "alpha": &alpha5, "fqi-hidden": &fqi16, "epsilon": &epsilon},
	}
	assert.Equal(t, expected, sw.allAttributes)
}
