package cem

import (
	"io"

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

// Debug sets a debug writer.
func Debug(opt io.Writer) Option {
	return optionFuncNil(func(cem *Cem) {
		cem.debugWriter = opt
	})
}
