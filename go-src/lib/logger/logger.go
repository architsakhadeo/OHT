package logger

import (
	"log"

	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

type LoggerConfig struct {
	// ShouldPrintDebug determines whether debug will be printed.
	ShouldPrintDebug bool

	// Interval is the number of steps between debug printout. If <0, will be considered 1.
	Interval int

	// ShouldLogData determines whether data is saved.
	ShouldLogData bool
}

// TODO Presumably the Logger prefixes a timestamp.
// TODO Logger could also eventually include debug levels.
// TODO Enable setting debug only for agent, environment, or experiment.
// TODO Allow it to load settings from the config.
// TODO The debug log and data log are really different things.
type Logger struct {
	LoggerConfig

	reward   []float64
	thisStep int
}

// New creates a new Logger. All debug uses the regular system log package.
//	-
//	- numberOfSteps is the number of steps to log. If 0, logging is disabled. Note if the number of steps is
//	  greater than this number, the additional data will still be logged, but memory allocations may occur.
func New(config LoggerConfig) rlglue.Logger {
	lg := &Logger{}
	lg.interval = interval
	if numberOfSteps != 0 {
		lg.shouldLog = true
	}
	lg.reward = make([]float64, 0, numberOfSteps)
	return lg
}

// Message logs a message followed by pairs of string-value to be stored in a structured log.
func (lg *Logger) Message(msg string, _ ...interface{}) {
	log.Println("Logger: " + msg)
}

// MessageDelta calls Message and appends the time since the last Message or MessageDelta.
func (lg *Logger) MessageDelta(msg string, _ ...interface{}) {
	log.Println("Logger: " + msg)
}

// MessageRewardSince calls Message and appends the reward since the provided step (calculated from the
// reward log).
func (lg *Logger) MessageRewardSince(step int, _ ...interface{}) {
	log.Printf("Logger: should print reward since step %d\n", step)
	// ep_return = np.sum(np.array(self.reward_log[self.old_count_total_step: numStepsTotal]))
	// print("\t\tEpisode {} ends, ep total step {}, ep return {}, accumulate reward {}\n".format(
	//     numEpisodes, self.step_log[-1], ep_return, np.sum(np.array(self.reward_log))))
}

// LogEpisodeLength adds the provided episode length to the episode length log.
func (lg *Logger) LogEpisodeLength(steps int) {
	log.Printf("Logger: episode length was %d\n", steps)
}

// LogStepHeader lists the headers used in the optional variadic arguments to LogStep.
func (lg *Logger) LogStepHeader(...string) {
	log.Println("Logger: some headers were added")
}

// LogStep adds information from a step to the step log. It must contain previous state, current state,
// and reward. It can optionally add other float64 values to be logged. (If so, LogStepHeader must be
// called to provide headers and so the logger knows how many to expect.)
func (lg *Logger) LogStep(prevState, currState rlglue.State, reward float64, others ...float64) {
	log.Println("Logger: some step data was saved")
	lg.reward[lg.thisStep] = reward
	// self.reward_log[numStepsTotal] = reward
	// self.trajectory_log[numStepsTotal, :self.dim_state] = s_t
	// self.trajectory_log[numStepsTotal, self.dim_state] = a_t
	// self.trajectory_log[numStepsTotal, self.dim_state+1: self.dim_state*2+1] = s_tp
	// self.trajectory_log[numStepsTotal, self.dim_state*2+1] = reward
	// self.trajectory_log[numStepsTotal, self.dim_state*2+2] = all the other ones
}

// Interval gives the desired number of steps to take between logging messages.
// This number is constant, so it should be cached for efficiency.
func (lg *Logger) Interval() int {
	if lg.interval <= 0 {
		return 1000000000 // A really large number, which should make the client log rarely
	}
	return lg.interval
}

// Save persists the logged information to disk.
func (lg *Logger) Save() {
	log.Println("Logger: log is being saved")
}
