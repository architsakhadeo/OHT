package state

import (
	"errors"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

// NewStateWrapperCreator is a function that can create a state wrapper.
type NewStateWrapperCreator func(logger.Debug, rlglue.Environment) (rlglue.Environment, error)

// Add is used to register a new state wrapper type.
// This is most likely called by an init function in the StateWrapper's go file.
// The function returns an error if a state wrapper with that name already exists.
func Add(name string, creator NewStateWrapperCreator) error {
	if _, ok := stateWrapperList[name]; ok {
		return errors.New("StateWrapper '" + name + "' has already been registered")
	}
	stateWrapperList[name] = creator
	return nil
}

func Create(name string, env rlglue.Environment, debug logger.Debug) (rlglue.Environment, error) {
	creator, ok := stateWrapperList[name]
	if !ok {
		return nil, errors.New("StateWrapper '" + name + "' has not been registered")
	}
	return creator(debug, env)
}

var stateWrapperList = map[string]NewStateWrapperCreator{}
