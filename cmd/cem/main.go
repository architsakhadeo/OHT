package main

import (
	"flag"
	"fmt"
	"runtime"

	"github.com/stellentus/cartpoles/lib/agent"
	"github.com/stellentus/cartpoles/lib/config"
	"github.com/stellentus/cartpoles/lib/environment"
	"github.com/stellentus/cartpoles/lib/experiment"
	"github.com/stellentus/cartpoles/lib/logger"
)

var (
	cpus = flag.Int("cpus", 2, "Number of CPUs")
)

func main() {
	flag.Parse()

	runtime.GOMAXPROCS(*cpus) // Limit the number of CPUs to the provided value (unchanged if the input is <1)

	debug := logger.NewDebug(logger.DebugConfig{}) // TODO create a debug

	for {
		agentSettings := agent.DefaultESarsaSettings()
		// Do CEM stuff to change settings and SEED

		ag := &agent.ESarsa{Debug: debug}
		ag.InitializeWithSettings(agentSettings)

		env := &environment.Cartpole{Debug: debug}
		env.InitializeWithSettings(environment.CartpoleSettings{Seed: int64(0)}) // TODO change seed

		data, err := logger.NewData(debug, logger.DataConfig{
			ShouldLogTraces:         false,
			CacheTracesInRAM:        false,
			ShouldLogEpisodeLengths: false,
			BasePath:                "",
			FileSuffix:              "",
		})
		panicIfError(err, "Couldn't create logger.Data")

		expConf := config.Experiment{
			MaxEpisodes:             10,
			MaxSteps:                0,
			DebugInterval:           0,
			DataPath:                "",
			ShouldLogTraces:         false,
			CacheTracesInRAM:        false,
			ShouldLogEpisodeLengths: false,
			MaxCPUs:                 *cpus,
		}
		exp, err := experiment.New(ag, env, expConf, debug, data)
		panicIfError(err, "Couldn't create experiment")

		exp.Run()

		fmt.Println(data.NumberOfEpisodes())
	}
}

func panicIfError(err error, reason string) {
	if err != nil {
		panic("ERROR " + err.Error() + ": " + reason)
	}
}
