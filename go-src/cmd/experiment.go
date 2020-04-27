package main

import (
	"encoding/json"
	"flag"
	"io/ioutil"

	"github.com/stellentus/cartpoles/go-src/lib/experiment"
	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

// Flags
var (
	configPath = flag.String("config", "", "config file for the experiment")
)

type Config struct {
	environment rlglue.Attributes
	agent       rlglue.Attributes
	experiment  json.RawMessage
}

func main() {
	data, err := ioutil.ReadFile(*configPath)
	if err != nil {
		panic("The config file at path '" + *configPath + "' could not be read")
	}

	var conf Config
	err = json.Unmarshal(data, &conf)
	if err != nil {
		panic("The config file at path '" + *configPath + "' is not valid JSON")
	}

	logger := logger.New()
	expr, err := experiment.New(conf.experiment, conf.agent, conf.environment, logger)
	expr.Run()
}
