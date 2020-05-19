package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/stellentus/cartpoles/lib/rlglue"
)

var Run = 0

type Config struct {
	EnvironmentName string            `json:"environment-name"`
	AgentName       string            `json:"agent-name"`
	Environment     rlglue.Attributes `json:"environment-settings"`
	Agent           rlglue.Attributes `json:"agent-settings"`
	Experiment      `json:"experiment-settings"`
	agentSweeper    Sweeper
	envSweeper      Sweeper
}

type Experiment struct {
	MaxEpisodes             int    `json:"episodes"`
	MaxSteps                int    `json:"steps"`
	DebugInterval           int    `json:"debug-interval"`
	DataPath                string `json:"data-path"`
	ShouldLogTraces         bool   `json:"should-log-traces"`
	CacheTracesInRAM        bool   `json:"cache-traces-in-ram"`
	ShouldLogEpisodeLengths bool   `json:"should-log-episode-lengths"`

	// MaxCPUs, if set, specifies the maximum number of CPUs this experiment is allowed to use
	MaxCPUs int `json:"max-cpus"`
}

func (set *Experiment) SetToDefault() {
	set.MaxEpisodes = 0
	set.MaxSteps = 0
	set.DebugInterval = 1
	set.DataPath = ""
	set.ShouldLogTraces = false
	set.ShouldLogEpisodeLengths = false
	set.CacheTracesInRAM = false
	set.MaxCPUs = 0 // Does not change the default value
}

// Parse parses a json.RawMessage. If the input is a JSON array, then that array as parsed as an array of config objects.
// Otherwise, it's parsed as a single config object.
func Parse(data json.RawMessage) ([]Config, error) {
	confJson := []json.RawMessage{}
	err := json.Unmarshal(data, &confJson)
	if err != nil {
		// Maybe it's a single conf, not an array
		conf, err := parseOne(data)
		if err != nil {
			return nil, err
		}
		return []Config{conf}, nil
	}

	confs := make([]Config, len(confJson))
	for i, cData := range confJson {
		confs[i], err = parseOne(cData)
		if err != nil {
			return nil, fmt.Errorf("Error parsing array element %d: %s", i, err.Error())
		}
	}

	return confs, nil
}

func parseOne(data json.RawMessage) (Config, error) {
	var conf Config
	conf.Experiment.SetToDefault()

	err := json.Unmarshal(data, &conf)
	if err != nil {
		return conf, errors.New("The config file is not valid JSON: " + err.Error())
	}

	err = conf.agentSweeper.Load(conf.Agent)
	if err != nil {
		return conf, errors.New("The agent sweeper could not be loaded: " + err.Error())
	}
	err = conf.envSweeper.Load(conf.Environment)
	if err != nil {
		return conf, errors.New("The environment sweeper could not be loaded: " + err.Error())
	}

	return conf, nil
}

func (conf Config) SweptAttrCount() int {
	// Assume we always sweep over at least one agent parameter.
	return len(conf.agentSweeper.allAttributes) * max(1, len(conf.envSweeper.allAttributes))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (conf Config) sweepIndices(idx int) (int, int, error) {
	agentCount := len(conf.agentSweeper.allAttributes)
	envCount := len(conf.envSweeper.allAttributes)
	agentIdx := idx % agentCount
	if envCount == 0 {
		return agentIdx, -1, nil
	}
	envIdx := idx / agentCount
	if envIdx >= envCount {
		return 0, 0, fmt.Errorf("The sweep idx is invalid")
	}
	return agentIdx, envIdx, nil
}

func (conf Config) SweptAttributes(idx int) (rlglue.Attributes, rlglue.Attributes, error) {
	totalCount := conf.SweptAttrCount()
	if idx >= totalCount {
		return nil, nil, fmt.Errorf("Cannot run sweep %d (max idx %d)", idx, totalCount-1)
	}
	agentIdx, envIdx, err := conf.sweepIndices(idx)
	if err != nil {
		return nil, nil, errors.New("Cannot run sweep: " + err.Error())
	}
	agentAttrs := conf.agentSweeper.allAttributes[agentIdx]
	agentAttributes, err := json.Marshal(agentAttrs)
	if err != nil {
		return nil, nil, fmt.Errorf("Cannot run agent sweep %d", agentIdx)
	}
	if envIdx == -1 {
		return agentAttributes, rlglue.Attributes(`{}`), nil
	}
	envAttrs := conf.envSweeper.allAttributes[envIdx]
	envAttributes, err := json.Marshal(envAttrs)
	if err != nil {
		return nil, nil, fmt.Errorf("Cannot run environment sweep %d", envIdx)
	}
	return agentAttributes, envAttributes, nil
}

type Sweeper struct {
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

func (swpr *Sweeper) Load(attributes rlglue.Attributes) error {
	swpr.allAttributes = []AttributeMap{}

	attrs := AttributeMap{}
	err := json.Unmarshal(attributes, &attrs)
	if err != nil {
		return errors.New("The attributes is not valid JSON: " + err.Error())
	}

	sweepAttrs, ok := attrs["sweep"]
	if !ok {
		return nil
	}
	delete(attrs, "sweep") // Neither Agent or Environment shouldn't receive the sweep info

	// Parse out the sweep arrays into key:array, where the array is still raw JSON.
	sweepRawJon := map[string]json.RawMessage{}
	err = json.Unmarshal(*sweepAttrs, &sweepRawJon)
	if err != nil {
		return errors.New("The swept attributes is not valid JSON: " + err.Error())
	}

	// Now for each key:array in JSON, convert the array to go arrays of raw JSON and count them.
	swpr.allAttributes = []AttributeMap{attrs}
	for key, val := range sweepRawJon {
		arrayVals := []json.RawMessage{}
		err = json.Unmarshal(val, &arrayVals)
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
