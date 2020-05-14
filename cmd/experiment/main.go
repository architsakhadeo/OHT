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
		sweepLen := conf.SweptAttrLen()
		for j := 0; j < sweepLen; j++ {
			fmt.Printf("Running sweep %d of %d\n", j+1, sweepLen)
			err = experiment.Execute(*run, conf, j)
		}
		if err != nil {
			panic("Could not create experiment: " + err.Error())
		}
	}
}
