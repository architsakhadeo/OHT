package logger

import (
	"log"
)

const longInterval = 1000000000 // A really large number, which should make the client log rarely

type DebugConfig struct {
	// ShouldPrintDebug determines whether debug will be printed.
	ShouldPrintDebug bool

	// Interval is the number of steps between debug printout. If <0, it will be considered 1.
	Interval int
}

type debugLogger struct {
	DebugConfig
}

// NewDebug creates a new logger.Debug. All debug uses the regular system log package.
func NewDebug(config DebugConfig) Debug {
	lg := &debugLogger{DebugConfig: config}
	if !config.ShouldPrintDebug {
		config.Interval = longInterval
	}
	return lg
}

// Message logs pairs of string-value to be stored in a structured log.
func (lg *debugLogger) Message(args ...interface{}) {
	if !lg.ShouldPrintDebug {
		return
	}
	log.Println(args)
}

// MessageDelta calls Message and appends the time since the last Message or MessageDelta.
func (lg *debugLogger) MessageDelta(args ...interface{}) {
	// TODO append delta-time
	lg.Message(args)
}

// Error logs an error if not nil.
func (lg *debugLogger) Error(err *error) {
	if *err != nil {
		lg.Message("err", (*err).Error())
	}
}

// Interval gives the desired number of steps to take between logging messages.
// This number is constant, so it should be cached for efficiency.
func (lg *debugLogger) Interval() int {
	if lg.DebugConfig.Interval <= 0 {
		return longInterval
	}
	return lg.DebugConfig.Interval
}
