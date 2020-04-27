package main

import (
	"encoding/json"
	"flag"
	"io/ioutil"

	_ "github.com/stellentus/cartpoles/go-src/lib/example"
	"github.com/stellentus/cartpoles/go-src/lib/experiment"
	"github.com/stellentus/cartpoles/go-src/lib/logger"
	_ "github.com/stellentus/cartpoles/go-src/lib/remote"
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

	debugLogger := logger.NewDebug(logger.DebugConfig{
		ShouldPrintDebug: true,
		Interval:         2,
	})
	dataLogger := logger.NewData(debugLogger, logger.DataConfig{
		ShouldLogTraces:         false,
		ShouldLogEpisodeLengths: true,
		NumberOfSteps:           1000,                     // TODO how to load this?
		BasePath:                "/save/here/from/config", // TODO
		FileSuffix:              "",                       // TODO after figuring out runs
	})

	expr, err := experiment.New(conf.Experiment, conf.Agent, conf.Environment, debugLogger, dataLogger)
	expr.Run()
}
