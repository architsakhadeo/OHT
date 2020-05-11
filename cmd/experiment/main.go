package main

import (
	"context"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/signal"
	"sync"

	"github.com/stellentus/cartpoles/lib/config"
	"github.com/stellentus/cartpoles/lib/experiment"
	"github.com/stellentus/cartpoles/lib/remote"
)

// Flags
var (
	configPath = flag.String("config", "config/example.json", "config file for the experiment")
	run        = flag.Uint("run", 0, "Run number")
)

func main() {
	flag.Parse()

	wg := &sync.WaitGroup{}
	ctx := context.Background()

	// trap Ctrl+C and call cancel on the context
	ctx, cancel := context.WithCancel(ctx)
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	defer func() {
		signal.Stop(c)
		cancel()
		wg.Wait()
	}()
	go func() {
		select {
		case <-c:
			cancel()
			wg.Wait()
			os.Exit(1)
		case <-ctx.Done():
		}
	}()
	remote.RegisterLaunchers(ctx, wg)

	data, err := ioutil.ReadFile(*configPath)
	if err != nil {
		panic("The config file at path '" + *configPath + "' could not be read: " + err.Error())
	}

	confs, err := config.Parse(data, int(*run))
	if err != nil {
		panic("Could not parse the config: " + err.Error())
	}

	for i, conf := range confs {
		if len(confs) > 1 {
			fmt.Printf("Running experiment %d of %d\n", i+1, len(confs))
		}
		err = experiment.Execute(*run, conf)
		if err != nil {
			panic("Could not create experiment: " + err.Error())
		}
	}
}
