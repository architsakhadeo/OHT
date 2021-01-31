package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"time"

	"github.com/stellentus/cartpoles/lib/cem"
)

var (
	configPath = flag.String("cem", "config/cem/cem.json", "CEM settings file path")
	expPath    = flag.String("exp", "config/cem/experiment.json", "Experiment settings file path")
	agentPath  = flag.String("agent", "config/cem/agent.json", "Default agent settings file path")
	envPath    = flag.String("env", "config/cem/environment.json", "Environment settings file path")
)

const e = 10.e-8

func main() {
	startTime := time.Now()
	flag.Parse()

	rn, err := cem.NewRunner(buildSettings())
	panicIfError(err, "Failed to create Runner")

	result, err := rn.Run([]cem.Option{cem.Debug(os.Stdout)})
	panicIfError(err, "Failed to run CEM")

	fmt.Println("\nFinal optional point: ", result)
	fmt.Println("Execution time: ", time.Since(startTime))
}

func panicIfError(err error, reason string) {
	if err != nil {
		panic("ERROR " + err.Error() + ": " + reason)
	}
}

func buildSettings() cem.RunnerSettings {
	// Build default settings
	settings := cem.RunnerSettings{
		Seed: math.MaxUint64,
	}
	settings.Settings = cem.DefaultSettings()

	readJsonFile(*configPath, &settings)
	readJsonFile(*expPath, &settings.ExperimentSettings)
	readJsonFile(*agentPath, &settings.AgentSettings)
	readJsonFile(*envPath, &settings.EnvironmentSettings)

	return settings
}

func readJsonFile(path string, val interface{}) {
	data, err := ioutil.ReadFile(path)
	panicIfError(err, "Couldn't load config file '"+path+"'")
	err = json.Unmarshal(data, val)
	panicIfError(err, "Couldn't parse config JSON '"+string(data)+"'")
}
