package logger

import (
	"fmt"
	"time"
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
	currentTime := time.Now()
	fmt.Print(currentTime.Format("2006/01/02 15:04:05 "))
	fmt.Println(args...)
}

// MessageDelta calls Message and appends the time since the last Message or MessageDelta.
func (lg *debugLogger) MessageDelta(args ...interface{}) {
	// TODO append delta-time
	lg.Message(args...)
}

// Error logs an error if not nil.
func (lg *debugLogger) Error(err *error) {
	if *err != nil {
		lg.Message("err", (*err).Error())
	}
}
