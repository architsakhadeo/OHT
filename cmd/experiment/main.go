package main

import (
	"flag"
	"fmt"
	"io/ioutil"

	"github.com/stellentus/cartpoles/lib/config"
	"github.com/stellentus/cartpoles/lib/experiment"
)

// Flags
var (
	configPath = flag.String("config", "config/esarsa.json", "config file for the experiment")
	run        = flag.Uint("run", 0, "Run number")
	sweep      = flag.Int("sweep", 0, "Sweep number")
	loadPath   = flag.String("load", "", "File to load agent from. (If not set, a new agent will be created.)")
	savePath   = flag.String("save", "", "File to save agent after the experement is complete. (If not set, nothing will be saved.)")
)

func main() {
	flag.Parse()

	data, err := ioutil.ReadFile(*configPath)
	if err != nil {
		panic("The config file at path '" + *configPath + "' could not be read: " + err.Error())
	}

	confs, err := config.Parse(data)
	if err != nil {
		panic("Could not parse the config: " + err.Error())
	}

	for i, conf := range confs {
		if len(confs) > 1 {
			fmt.Printf("Running experiment %d of %d\n", i+1, len(confs))
		}
		sweepLen := conf.SweptAttrCount()
		if *sweep >= sweepLen {
			panic(fmt.Sprintf("Could not run sweep %d (range should be 0 to %d)", *sweep, sweepLen-1))
		}

		if *sweep != 0 {
			fmt.Printf("Running sweep %d of %d\n", *sweep, sweepLen)
		}
		err = experiment.Execute(*run, conf, *sweep, *savePath, *loadPath)

		if err != nil {
			panic("Could not create experiment: " + err.Error())
		}
	}
}
