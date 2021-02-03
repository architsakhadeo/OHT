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
	sweep      = flag.Int("sweep", 0, "Sweep number (-1 to run all)")
)

func main() {
	flag.Parse()

	data, err := ioutil.ReadFile(*configPath)
	panicIfError(err, "The config file at path '"+*configPath+"' could not be read")

	confs, err := config.Parse(data)
	panicIfError(err, "Could not parse the config")

	for i, conf := range confs {
		if len(confs) > 1 {
			fmt.Printf("Running experiment %d of %d\n", i+1, len(confs))
		}

		sweepLen := conf.SweptAttrCount()
		if *sweep == -1 {
			fmt.Println("Running ALL sweeps...")
			for sw := 0; sw < sweepLen; sw++ {
				runSweepN(*run, conf, sw, sweepLen)
			}
		} else {
			if *sweep >= sweepLen {
				panic(fmt.Sprintf("Could not run sweep %d (range should be 0 to %d)", *sweep, sweepLen-1))
			}
			runSweepN(*run, conf, *sweep, sweepLen)
		}
	}
}

func runSweepN(run uint, conf config.Config, sweep, sweepLen int) {
	if sweep != 0 {
		fmt.Printf("Running sweep %d of %d\n", sweep, sweepLen)
	}
	err := experiment.Execute(run, conf, sweep)
	panicIfError(err, "Could not create the experiment")
}

func panicIfError(err error, reason string) {
	if err != nil {
		panic("ERROR " + err.Error() + ": " + reason)
	}
}
