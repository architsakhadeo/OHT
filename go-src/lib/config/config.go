package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

var Run = 0

type Config struct {
	EnvironmentName string            `json:"environment-name"`
	AgentName       string            `json:"agent-name"`
	Environment     rlglue.Attributes `json:"environment-settings"`
	Agent           rlglue.Attributes `json:"agent-settings"`
	Experiment      `json:"experiment-settings"`
	Run             int `json:"run"`
	sweeper
}

type Experiment struct {
	MaxEpisodes             int    `json:"episodes"`
	MaxSteps                int    `json:"steps"`
	DebugInterval           int    `json:"debug-interval"`
	DataPath                string `json:"data-path"`
	ShouldLogTraces         bool   `json:"should-log-traces"`
	ShouldLogEpisodeLengths bool   `json:"should-log-episode-lengths"`
}

func (set *Experiment) SetToDefault() {
	set.MaxEpisodes = 0
	set.MaxSteps = 0
	set.DebugInterval = 1
	set.DataPath = ""
	set.ShouldLogTraces = false
	set.ShouldLogEpisodeLengths = false
}

// If run>=0, it's used to override the value in the config. If it's also not set in the config, it's 0.
func Parse(data json.RawMessage, run int) (Config, error) {
	var conf Config
	conf.Experiment.SetToDefault()

	err := json.Unmarshal(data, &conf)
	if err != nil {
		return conf, errors.New("The config file is not valid JSON: " + err.Error())
	}

	if run >= 0 {
		conf.Run = run
	}
	Run = conf.Run

	err = conf.LoadSweeper()
	if err != nil {
		return conf, errors.New("The sweeper could not be loaded: " + err.Error())
	}

	return conf, nil
}

func (conf Config) SweptAgentAttributes(idx int) (rlglue.Attributes, error) {
	if idx >= len(conf.sweeper.allAttributes) {
		return nil, fmt.Errorf("Cannot run sweep %d (max idx %d)", idx, len(conf.sweeper.allAttributes)-1)
	}
	attrs := conf.sweeper.allAttributes[idx]
	return json.Marshal(attrs)
}

type sweeper struct {
	allAttributes []AttributeMap
}

type AttributeMap map[string]*json.RawMessage

func (am AttributeMap) String() string {
	strs := []string{}
	for key, val := range am {
		strs = append(strs, fmt.Sprintf("%s:%v", key, string(*val)))
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

func (conf *Config) LoadSweeper() error {
	conf.sweeper.allAttributes = []AttributeMap{}

	agentAttrs := AttributeMap{}
	err := json.Unmarshal(conf.Agent, &agentAttrs)
	if err != nil {
		return errors.New("The agent attributes is not valid JSON: " + err.Error())
	}

	sweepAttrs, ok := agentAttrs["sweep"]
	if !ok {
		return nil
	}
	delete(agentAttrs, "sweep") // Agent shouldn't receive the sweep info

	// Parse out the sweep arrays into key:array, where the array is still raw JSON.
	sweepRawJon := map[string]json.RawMessage{}
	err = json.Unmarshal(*sweepAttrs, &sweepRawJon)
	if err != nil {
		return errors.New("The agent attributes is not valid JSON: " + err.Error())
	}

	// Now for each key:array in JSON, convert the array to go arrays of raw JSON and count them.
	conf.sweeper.allAttributes = []AttributeMap{agentAttrs}
	for key, val := range sweepRawJon {
		arrayVals := []json.RawMessage{}
		err = json.Unmarshal(val, &arrayVals)
		if err != nil {
			return errors.New("The agent attributes is not valid JSON: " + err.Error())
		}
		if len(arrayVals) == 0 {
			break // This array is empty, so nothing to do here
		}

		newAMSlice := []AttributeMap{}
		for _, am := range conf.sweeper.allAttributes {
			for i, av := range arrayVals {
				newAM := am
				if i != 0 {
					// For the first new value, we can use the previous one instead of copying. All others must copy.
					newAM = am.Copy()
				}
				newAM[key] = &av
				newAMSlice = append(newAMSlice, newAM)
			}
		}
		conf.sweeper.allAttributes = newAMSlice
	}

	return nil
}
