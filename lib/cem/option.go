package cem

import (
	"errors"

	"github.com/stellentus/cartpoles/lib/logger"
	"golang.org/x/exp/rand"
)

type Option interface {
	apply(*Cem) error
}

// optionFunc wraps a func so it satisfies the Option interface.
type optionFunc func(*Cem) error
type optionFuncNil func(*Cem)

func (f optionFunc) apply(cem *Cem) error    { return f(cem) }
func (f optionFuncNil) apply(cem *Cem) error { f(cem); return nil }

// Seed sets the seed.
// If not set, the seed defaults to the current time.
func Seed(opt uint64) Option {
	return optionFuncNil(func(cem *Cem) {
		cem.rng = rand.New(rand.NewSource(opt))
	})
}

// NumWorkers is the maximum number of workers. Must be at least 1.
// If not set, defaults to runtime.NumCPUs.
func NumWorkers(opt int) Option {
	return optionFunc(func(cem *Cem) error {
		if cem.numWorkers <= 0 {
			return errors.New("Number of workers must be at least 1")
		}
		cem.numWorkers = opt
		return nil
	})
}

// NumIterations is the total number of iterations
func NumIterations(opt int) Option {
	return optionFuncNil(func(cem *Cem) {
		cem.numIterations = opt
	})
}

// NumSamples is the number of samples per iteration
func NumSamples(opt int) Option {
	return optionFuncNil(func(cem *Cem) {
		cem.numSamples = opt
	})
}

// NumRuns is the number of runs per sample
func NumRuns(opt int) Option {
	return optionFuncNil(func(cem *Cem) {
		cem.numRuns = opt
	})
}

// NumTimesteps is the number of timesteps per run
func NumTimesteps(opt int) Option {
	return optionFuncNil(func(cem *Cem) {
		cem.numTimesteps = opt
	})
}

// NumEpisodes is the number of episodes
func NumEpisodes(opt int) Option {
	return optionFuncNil(func(cem *Cem) {
		cem.numEpisodes = opt
	})
}

// NumStepsInEpisode is the number of steps in episode
func NumStepsInEpisode(opt int) Option {
	return optionFuncNil(func(cem *Cem) {
		cem.numStepsInEpisode = opt
	})
}

// MaxRunLengthEpisodic is the max number of steps in episode
func MaxRunLengthEpisodic(opt int) Option {
	return optionFuncNil(func(cem *Cem) {
		cem.maxRunLengthEpisodic = opt
	})
}

// MaxEpisodes is the max number of episodes in an experiment
func MaxEpisodes(opt int) Option {
	return optionFuncNil(func(cem *Cem) {
		cem.maxEpisodes = opt
	})
}

// PercentElite is the percent of samples that should be drawn from the elite group
func PercentElite(opt float64) Option {
	return optionFuncNil(func(cem *Cem) {
		cem.percentElite = opt
	})
}

// DebugLogger sets the debug logger.
func DebugLogger(opt logger.Debug) Option {
	return optionFuncNil(func(cem *Cem) {
		cem.debug = opt
	})
}

// DataLogger configures the agent settings.
func DataLogger(opt logger.Data) Option {
	return optionFuncNil(func(cem *Cem) {
		cem.data = opt
	})
}
