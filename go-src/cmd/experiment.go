package main

import (
	"encoding/json"
	"flag"
	"io/ioutil"

	_ "github.com/stellentus/cartpoles/go-src/lib/example"
	"github.com/stellentus/cartpoles/go-src/lib/experiment"
	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

// Flags
var (
	configPath = flag.String("config", "config/example.json", "config file for the experiment")
)

type Config struct {
	Environment rlglue.Attributes `json:"environment"`
	Agent       rlglue.Attributes `json:"agent"`
	Experiment  json.RawMessage   `json:"experiment"`
}

func main() {
	flag.Parse()

	data, err := ioutil.ReadFile(*configPath)
	if err != nil {
		panic("The config file at path '" + *configPath + "' could not be read: " + err.Error())
	}

	var conf Config
	err = json.Unmarshal(data, &conf)
	if err != nil {
		panic("The config file at path '" + *configPath + "' is not valid JSON: " + err.Error())
	}

	logger := logger.New()
	expr, err := experiment.New(conf.Experiment, conf.Agent, conf.Environment, logger)
	expr.Run()
}
