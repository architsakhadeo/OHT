package config

import (
	"encoding/json"
	"errors"

	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

type Config struct {
	EnvironmentName string            `json:"environment-name"`
	AgentName       string            `json:"agent-name"`
	Environment     rlglue.Attributes `json:"environment-settings"`
	Agent           rlglue.Attributes `json:"agent-settings"`
	Experiment      `json:"experiment-settings"`
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

func Parse(data json.RawMessage) (Config, error) {
	var conf Config
	conf.Experiment.SetToDefault()

	err := json.Unmarshal(data, &conf)
	if err != nil {
		return conf, errors.New("The config file is not valid JSON: " + err.Error())
	}

	return conf, nil
}
