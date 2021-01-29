package cem

import (
	"errors"

	"github.com/stellentus/cartpoles/lib/logger"
	"golang.org/x/exp/rand"
)

type Option func(*Cem) error

// Seed sets the seed.
// If not set, the seed defaults to the current time.
func Seed(opt uint64) Option {
	return func(cem *Cem) error {
		cem.rng = rand.New(rand.NewSource(opt))
		return nil
	}
}

// NumWorkers is the maximum number of workers. Must be at least 1.
// If not set, defaults to runtime.NumCPUs.
func NumWorkers(opt int) Option {
	return func(cem *Cem) error {
		if cem.numWorkers <= 0 {
			return errors.New("Number of workers must be at least 1")
		}
		cem.numWorkers = opt
		return nil
	}
}

// NumIterations is the total number of iterations
func NumIterations(opt int) Option {
	return func(cem *Cem) error {
		cem.numIterations = opt
		return nil
	}
}

// NumSamples is the number of samples per iteration
func NumSamples(opt int) Option {
	return func(cem *Cem) error {
		cem.numSamples = opt
		return nil
	}
}

// NumRuns is the number of runs per sample
func NumRuns(opt int) Option {
	return func(cem *Cem) error {
		cem.numRuns = opt
		return nil
	}
}

// NumTimesteps is the number of timesteps per run
func NumTimesteps(opt int) Option {
	return func(cem *Cem) error {
		cem.numTimesteps = opt
		return nil
	}
}

// NumEpisodes is the number of episodes
func NumEpisodes(opt int) Option {
	return func(cem *Cem) error {
		cem.numEpisodes = opt
		return nil
	}
}

// NumStepsInEpisode is the number of steps in episode
func NumStepsInEpisode(opt int) Option {
	return func(cem *Cem) error {
		cem.numStepsInEpisode = opt
		return nil
	}
}

// MaxRunLengthEpisodic is the max number of steps in episode
func MaxRunLengthEpisodic(opt int) Option {
	return func(cem *Cem) error {
		cem.maxRunLengthEpisodic = opt
		return nil
	}
}

// PercentElite is the percent of samples that should be drawn from the elite group
func PercentElite(opt float64) Option {
	return func(cem *Cem) error {
		cem.percentElite = opt
		return nil
	}
}

// DebugLogger sets the debug logger.
func DebugLogger(opt logger.Debug) Option {
	return func(cem *Cem) error {
		cem.debug = opt
		return nil
	}
}
