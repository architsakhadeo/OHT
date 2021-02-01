package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/stellentus/cartpoles/lib/rlglue"
)

type sweeper struct {
	allAttributes []AttributeMap
}

type AttributeMap map[string]*json.RawMessage

func (am AttributeMap) String() string {
	keys := make([]string, 0, len(am))
	for key, _ := range am {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	strs := make([]string, len(am))
	for i, key := range keys {
		val := am[key]
		strs[i] = fmt.Sprintf("%s:%v", key, string(*val))
	}
	return "AM<" + strings.Join(strs, ", ") + ">"
}
func (am AttributeMap) Copy() AttributeMap {
	am2 := AttributeMap{}
	for key, val := range am {
		am2[key] = val
	}
	return am2
}

func (swpr *sweeper) Load(attributes rlglue.Attributes) error {
	swpr.allAttributes = []AttributeMap{}

	attrs := AttributeMap{}
	err := json.Unmarshal(attributes, &attrs)
	if err != nil {
		return errors.New("The attributes is not valid JSON: " + err.Error())
	}
	swpr.allAttributes = []AttributeMap{attrs}
	sweepAttrs, ok := attrs["sweep"]
	if !ok {
		return nil
	}
	delete(attrs, "sweep") // Neither Agent or Environment shouldn't receive the sweep info

	typeStr := "sweep" // default type
	sweepType, ok := attrs["sweep-type"]
	if ok {
		delete(attrs, "sweep-type")
		err = json.Unmarshal(*sweepType, &typeStr)
		if err != nil {
			return errors.New("The sweep type is not valid JSON: " + err.Error())
		}
	}

	switch typeStr {
	case "sweep":
		return swpr.loadSweeps(sweepAttrs)
	case "list":
		return swpr.loadList(sweepAttrs)
	default:
		return errors.New("Could not load unknown sweep type '" + typeStr + "'")
	}
}

func (swpr *sweeper) loadSweeps(sweepAttrs *json.RawMessage) error {
	// Parse out the sweep arrays into key:array, where the array is still raw JSON.
	sweepRawJson := map[string]json.RawMessage{}
	err := json.Unmarshal(*sweepAttrs, &sweepRawJson)
	if err != nil {
		return errors.New("The swept attributes is not valid JSON: " + err.Error())
	}

	// Now for each key:array in JSON, convert the array to go arrays of raw JSON and count them.
	// Sort keys for reproducibility.
	keys := make([]string, 0)
	for key := range sweepRawJson {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		arrayVals := []json.RawMessage{}
		err = json.Unmarshal(sweepRawJson[key], &arrayVals)
		if err != nil {
			return errors.New("The attributes is not valid JSON: " + err.Error())
		}
		if len(arrayVals) == 0 {
			break // This array is empty, so nothing to do here
		}

		newAMSlice := []AttributeMap{}
		for _, am := range swpr.allAttributes {
			for i := range arrayVals {
				newAM := am
				if i != 0 {
					// For the first new value, we can use the previous one instead of copying. All others must copy.
					newAM = am.Copy()
				}
				av := arrayVals[i]
				newAM[key] = &av
				newAMSlice = append(newAMSlice, newAM)
			}
		}
		swpr.allAttributes = newAMSlice
	}

	return nil
}

func (swpr *sweeper) loadList(sweepAttrs *json.RawMessage) error {
	// Parse out the array of settings.
	sweepRawJson := []json.RawMessage{}
	err := json.Unmarshal(*sweepAttrs, &sweepRawJson)
	if err != nil {
		return errors.New("The swept attributes is not valid JSON: " + err.Error())
	}

	listOfHypers := []map[string]json.RawMessage{}
	for _, hypers := range sweepRawJson {
		namedHypers := map[string]json.RawMessage{}
		err = json.Unmarshal(hypers, &namedHypers)
		if err != nil {
			return errors.New("The attributes is not valid JSON: " + err.Error())
		}
		if len(namedHypers) == 0 {
			break // This array is empty, so nothing to do here
		}
		listOfHypers = append(listOfHypers, namedHypers)
	}

	newAMSlice := []AttributeMap{}
	for _, am := range swpr.allAttributes {
		newAM := am
		for _, namedHypers := range listOfHypers {
			newAM = am.Copy()
			for key := range namedHypers {
				val := namedHypers[key]
				newAM[key] = &val
			}
			newAMSlice = append(newAMSlice, newAM)
		}
	}
	swpr.allAttributes = newAMSlice

	return nil
}
