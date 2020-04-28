package main

import (
	"flag"
	"io/ioutil"

	"github.com/stellentus/cartpoles/go-src/lib/config"
	"github.com/stellentus/cartpoles/go-src/lib/experiment"
	_ "github.com/stellentus/cartpoles/go-src/lib/remote"
)

// Flags
var (
	configPath = flag.String("config", "config/example.json", "config file for the experiment")
)

func main() {
	flag.Parse()

	data, err := ioutil.ReadFile(*configPath)
	if err != nil {
		panic("The config file at path '" + *configPath + "' could not be read: " + err.Error())
	}

	conf, err := config.Parse(data)
	if err != nil {
		panic("Could not parse the config: " + err.Error())
	}

	err = experiment.Execute(conf)
	if err != nil {
		panic("Could not create the experiment: " + err.Error())
	}
}
