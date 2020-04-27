package logger

import "github.com/stellentus/cartpoles/go-src/lib/rlglue"

// TODO Presumably the Logger prefixes a timestamp.
// TODO Logger could also eventually include debug levels.
// TODO Enable setting debug only for agent, environment, or experiment.
// TODO Allow it to load settings from the config.
type Logger struct {
}

func New() rlglue.Logger {
	return &Logger{}
}

// Message logs a message followed by pairs of string-value to be stored in a structured log.
func (lg *Logger) Message(string, ...interface{}) {
	panic("logger.Logger's Message is not implemented")
}

// MessageDelta calls Message and appends the time since the last Message or MessageDelta.
func (lg *Logger) MessageDelta(string, ...interface{}) {
	panic("logger.Logger's MessageDelta is not implemented")
}

// MessageRewardSince calls Message and appends the reward since the provided step (calculated from the
// reward log).
func (lg *Logger) MessageRewardSince(int, ...interface{}) {
	panic("logger.Logger's MessageRewardSince is not implemented")
}

// LogEpisodeLength adds the provided episode length to the episode length log.
func (lg *Logger) LogEpisodeLength(int) {
	panic("logger.Logger's LogEpisodeLength is not implemented")
}

// LogStepHeader lists the headers used in the optional variadic arguments to LogStep.
func (lg *Logger) LogStepHeader(...string) {
	panic("logger.Logger's LogStepHeader is not implemented")
}

// LogStep adds information from a step to the step log. It must contain previous state, current state,
// and reward. It can optionally add other float64 values to be logged. (If so, LogStepHeader must be
// called to provide headers and so the logger knows how many to expect.)
func (lg *Logger) LogStep(rlglue.State, rlglue.State, float64, ...float64) {
	panic("logger.Logger's LogStep is not implemented")
}

// Interval gives the desired number of steps to take between logging messages.
// This number is constant, so it should be cached for efficiency.
func (lg *Logger) Interval() int {
	panic("logger.Logger's Interval is not implemented")
}

// Save persists the logged information to disk.
func (lg *Logger) Save() {
	panic("logger.Logger's Save is not implemented")
}
