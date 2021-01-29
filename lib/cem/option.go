package cem

import (
	"errors"

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

// PercentElite is the percent of samples that should be drawn from the elite group
func PercentElite(opt float64) Option {
	return optionFuncNil(func(cem *Cem) {
		cem.percentElite = opt
	})
}
