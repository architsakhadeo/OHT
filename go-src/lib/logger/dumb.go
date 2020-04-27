package logger

import (
	"log"

	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

// TODO Presumably the Logger prefixes a timestamp.
// TODO Logger could also eventually include debug levels.
// TODO Enable setting debug only for agent, environment, or experiment.
// TODO Allow it to load settings from the config.
// TODO The debug log and data log are really different things.
type DumbLog struct {
}

func New() rlglue.Logger {
	return &DumbLog{}
}

// Message logs a message followed by pairs of string-value to be stored in a structured log.
func (lg *DumbLog) Message(msg string, _ ...interface{}) {
	log.Println("DumbLog: " + msg)
}

// MessageDelta calls Message and appends the time since the last Message or MessageDelta.
func (lg *DumbLog) MessageDelta(msg string, _ ...interface{}) {
	log.Println("DumbLog: " + msg)
}

// MessageRewardSince calls Message and appends the reward since the provided step (calculated from the
// reward log).
func (lg *DumbLog) MessageRewardSince(step int, _ ...interface{}) {
	log.Printf("DumbLog: should print reward since step %d\n", step)
	// ep_return = np.sum(np.array(self.reward_log[self.old_count_total_step: numStepsTotal]))
	// print("\t\tEpisode {} ends, ep total step {}, ep return {}, accumulate reward {}\n".format(
	//     numEpisodes, self.step_log[-1], ep_return, np.sum(np.array(self.reward_log))))
}

// LogEpisodeLength adds the provided episode length to the episode length log.
func (lg *DumbLog) LogEpisodeLength(steps int) {
	log.Printf("DumbLog: episode length was %d\n", steps)
}

// LogStepHeader lists the headers used in the optional variadic arguments to LogStep.
func (lg *DumbLog) LogStepHeader(...string) {
	log.Println("DumbLog: some headers were added")
}

// LogStep adds information from a step to the step log. It must contain previous state, current state,
// and reward. It can optionally add other float64 values to be logged. (If so, LogStepHeader must be
// called to provide headers and so the logger knows how many to expect.)
func (lg *DumbLog) LogStep(rlglue.State, rlglue.State, float64, ...float64) {
	log.Println("DumbLog: some step data was saved")
	// self.reward_log[numStepsTotal] = reward
	// self.trajectory_log[numStepsTotal, :self.dim_state] = s_t
	// self.trajectory_log[numStepsTotal, self.dim_state] = a_t
	// self.trajectory_log[numStepsTotal, self.dim_state+1: self.dim_state*2+1] = s_tp
	// self.trajectory_log[numStepsTotal, self.dim_state*2+1] = reward
	// self.trajectory_log[numStepsTotal, self.dim_state*2+2] = all the other ones
}

// Interval gives the desired number of steps to take between logging messages.
// This number is constant, so it should be cached for efficiency.
func (lg *DumbLog) Interval() int { return 1 }

// Save persists the logged information to disk.
func (lg *DumbLog) Save() {
	log.Println("DumbLog: log is being saved")
}
