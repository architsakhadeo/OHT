package config

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/stellentus/cartpoles/lib/rlglue"
)

var Run = 0

type Config struct {
	EnvironmentName string            `json:"environment-name"`
	AgentName       string            `json:"agent-name"`
	Environment     rlglue.Attributes `json:"environment-settings"`
	Agent           rlglue.Attributes `json:"agent-settings"`
	Experiment      `json:"experiment-settings"`
	StateWrappers   rlglue.Attributes `json:"state-wrappers"`
	agentSweeper    sweeper
	envSweeper      sweeper
	wrapperSweepers []sweeper
	WrapperNames    []string
}

type Experiment struct {
	MaxEpisodes                   int    `json:"episodes"`
	MaxSteps                      int    `json:"steps"`
	MaxStepsInEpisode             int    `json:"steps-in-episode"`
	MaxRunLengthEpisodic          int    `json:"max-run-length-episodic"`
	TrainingStepsBeforeLock       int    `json:"training-steps-before-lock"`
	DebugInterval                 int    `json:"debug-interval"`
	DataPath                      string `json:"data-path"`
	ShouldLogTraces               bool   `json:"should-log-traces"`
	CacheTracesInRAM              bool   `json:"cache-traces-in-ram"`
	ShouldLogEpisodeLengths       bool   `json:"should-log-episode-lengths"`
	ShouldLogReturns              bool   `json:"should-log-returns"`
	ShouldLogRewards              bool   `json:"should-log-rewards"`
	ShouldLogTotals               bool   `json:"should-log-totals"`
	CountAfterLock                bool   `json:"count-step-after-lock"`
	TotalAfterCount               int    `json:"total-step-after-lock"`
	RandomizeStartStateBeforeLock bool   `json:"randomize_start_state_beforeLock"`
	RandomizeStartStateAfterLock  bool   `json:"randomize_start_state_afterLock"`
	ShouldLogLearnProg            bool   `json:"should-log-learn-progress"`
	// MaxCPUs, if set, specifies the maximum number of CPUs this experiment is allowed to use
	MaxCPUs int `json:"max-cpus"`
}

func (set *Experiment) SetToDefault() {
	set.MaxEpisodes = 0
	set.MaxSteps = 0
	set.MaxStepsInEpisode = -1
	set.MaxRunLengthEpisodic = -1
	set.DebugInterval = 1
	set.DataPath = ""
	set.ShouldLogTraces = false
	set.ShouldLogEpisodeLengths = false
	set.ShouldLogReturns = false
	set.ShouldLogLearnProg = false
	set.CacheTracesInRAM = false
	set.MaxCPUs = 0 // Does not change the default value
	set.CountAfterLock = false
	set.TotalAfterCount = 0
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

type AttributeMapAttr map[string]*rlglue.Attributes

func parseOne(data json.RawMessage) (Config, error) {
	var conf Config
	conf.Experiment.SetToDefault()

	err := json.Unmarshal(data, &conf)
	if err != nil {
		return conf, errors.New("The config file is not valid JSON: " + err.Error())
	}

	if conf.StateWrappers != nil {
		wrapperAttrs := []AttributeMapAttr{}
		err = json.Unmarshal(conf.StateWrappers, &wrapperAttrs)
		if err != nil {
			return conf, errors.New("The attributes is not valid JSON: " + err.Error())
		}
		for _, wrapperAttr := range wrapperAttrs {
			wrapperSwpr := sweeper{}
			wrapperSwpr.Load(*wrapperAttr["settings"])
			var wrapperName string
			err := json.Unmarshal(*wrapperAttr["wrapper-name"], &wrapperName)
			if err != nil {
				return conf, errors.New("The wrapper attribute is not valid JSON: " + err.Error())
			}
			conf.WrapperNames = append(conf.WrapperNames, wrapperName)
			conf.wrapperSweepers = append(conf.wrapperSweepers, wrapperSwpr)
		}
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
	// Assume there is at least one parameter in the agent/environment/wrapper setting.
	count := len(conf.agentSweeper.allAttributes) * len(conf.envSweeper.allAttributes)
	for _, wrapperSwpr := range conf.wrapperSweepers {
		count = count * len(wrapperSwpr.allAttributes)
	}
	return count
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (conf Config) sweepIndices(idx int) ([]int, error) {
	totalCount := conf.SweptAttrCount()
	agentCount := len(conf.agentSweeper.allAttributes)
	envCount := len(conf.envSweeper.allAttributes)
	counts := []int{agentCount, envCount}
	for _, wrapperSwpr := range conf.wrapperSweepers {
		counts = append(counts, len(wrapperSwpr.allAttributes))
	}
	indices := []int{}
	for i := len(counts) - 1; i >= 0; i-- {
		totalCount = totalCount / counts[i]
		q := idx / totalCount
		indices = append([]int{q}, indices...)
		idx = idx % totalCount
		if i == 1 {
			indices = append([]int{idx}, indices...)
			break
		}
	}
	return indices, nil
}

func (conf Config) SweptAttributes(idx int) ([]rlglue.Attributes, error) {
	totalCount := conf.SweptAttrCount()
	if idx >= totalCount {
		return nil, fmt.Errorf("Cannot run sweep %d (max idx %d)", idx, totalCount-1)
	}
	indices, err := conf.sweepIndices(idx)
	agentIdx, envIdx, wrapperIds := indices[0], indices[1], indices[2:]
	if err != nil {
		return nil, errors.New("Cannot run sweep: " + err.Error())
	}
	agentAttrs := conf.agentSweeper.allAttributes[agentIdx]
	agentAttributes, err := json.Marshal(agentAttrs)
	if err != nil {
		return nil, fmt.Errorf("Cannot run agent sweep %d", agentIdx)
	}
	envAttrs := conf.envSweeper.allAttributes[envIdx]
	envAttributes, err := json.Marshal(envAttrs)
	if err != nil {
		return nil, fmt.Errorf("Cannot run environment sweep %d", envIdx)
	}
	attributes := []rlglue.Attributes{agentAttributes, envAttributes}
	for i, wrapperSwpr := range conf.wrapperSweepers {
		wrapperAttrs := wrapperSwpr.allAttributes[wrapperIds[i]]
		wrapperAttributes, err := json.Marshal(wrapperAttrs)
		if err != nil {
			return nil, fmt.Errorf("Cannot run wrapper sweep %d", wrapperIds[i])
		}
		attributes = append(attributes, wrapperAttributes)
	}
	return attributes, nil
}
