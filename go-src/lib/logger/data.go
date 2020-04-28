package logger

import (
	"fmt"
	"os"
	"path"

	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

type DataConfig struct {
	// ShouldLogTraces determines whether traces are saved (reward, state, prevState, action, and any
	// other provided values). Rewards are always recorded.
	ShouldLogTraces bool

	// ShouldLogEpisodeLengths determines whether episode lengths saved.
	ShouldLogEpisodeLengths bool

	// BasePath is the path at which files are saved. A filename will be automatically set (rewards, traces, and episodes).
	// If not set, no file is saved.
	BasePath string

	// FileSuffix is a suffix for the filename (most often used for run numbers). A dash will automatically be inserted.
	FileSuffix string
}

type dataLogger struct {
	Debug

	DataConfig

	episodeLengths []int

	prevState []rlglue.State
	currState []rlglue.State
	actions   []rlglue.Action
	rewards   []float64
	others    [][]float64

	headers []string
}

func NewData(debug Debug, config DataConfig) (Data, error) {
	lg := &dataLogger{
		Debug:      debug,
		DataConfig: config,
		rewards:    []float64{},
	}
	if lg.ShouldLogTraces {
		lg.prevState = []rlglue.State{}
		lg.currState = []rlglue.State{}
		lg.actions = []rlglue.Action{}
	}
	if lg.ShouldLogEpisodeLengths {
		lg.episodeLengths = []int{}
	}
	var err error
	if lg.BasePath != "" {
		err = os.MkdirAll(lg.BasePath, os.ModePerm) // Ensure the directory exists (TODO ensure it's writable)
	}
	return lg, err
}

func (lg *dataLogger) RewardSince(step int) float64 {
	var sum float64
	end := len(lg.rewards)
	for i := step; i < end; i++ {
		sum += lg.rewards[i]
	}
	return sum
}

// LogEpisodeLength adds the provided episode length to the episode length log.
func (lg *dataLogger) LogEpisodeLength(steps int) {
	if !lg.ShouldLogEpisodeLengths {
		return
	}
	lg.episodeLengths = append(lg.episodeLengths, steps)
}

// LogStepHeader lists the headers used in the optional variadic arguments to LogStep.
func (lg *dataLogger) LogStepHeader(headers ...string) {
	if lg.headers != nil {
		lg.Message("err", "Attempt to add headers after steps have been recorded", "steps", len(lg.rewards), "headers", headers)
		return
	}
	for _, hdr := range headers {
		lg.headers = append(lg.headers, hdr)
		lg.others = append(lg.others, []float64{})
	}
}

// LogStep adds information from a step to the step log. It must contain previous state, current state,
// and reward. It can optionally add other float64 values to be logged. (If so, LogStepHeader must be
// called to provide headers and so the logger knows how many to expect.)
func (lg *dataLogger) LogStep(prevState, currState rlglue.State, action rlglue.Action, reward float64) {
	lg.rewards = append(lg.rewards, reward)

	if lg.ShouldLogTraces {
		lg.prevState = append(lg.prevState, prevState)
		lg.currState = append(lg.currState, currState)
		lg.actions = append(lg.actions, action)
	}
}

// LogStepMulti is like LogStep, but it can optionally add other float64 values to be logged. (If so,
// LogStepHeader must be called to provide headers and so the logger knows how many to expect.)
func (lg *dataLogger) LogStepMulti(prevState, currState rlglue.State, action rlglue.Action, reward float64, others ...float64) {
	if lg.ShouldLogTraces {
		for i, other := range others {
			lg.others[i] = append(lg.others[i], other)
		}
	}
	lg.LogStep(prevState, currState, action, reward)
}

// Save persists the logged information to disk.
func (lg *dataLogger) SaveLog() error {
	if lg.BasePath == "" {
		return nil
	}

	file, err := os.Create(path.Join(lg.BasePath, "rewards.csv", lg.FileSuffix))
	if err != nil {
		return err
	}
	for _, rew := range lg.rewards {
		_, err = file.WriteString(fmt.Sprintf("%f\n", rew))
		if err != nil {
			return err
		}
	}

	if lg.ShouldLogEpisodeLengths {
		file, err := os.Create(path.Join(lg.BasePath, "episodes.csv", lg.FileSuffix))
		if err != nil {
			return err
		}
		for _, ep := range lg.episodeLengths {
			_, err = file.WriteString(fmt.Sprintf("%d\n", ep))
			if err != nil {
				return err
			}
		}
	}

	if lg.ShouldLogTraces {
		file, err := os.Create(path.Join(lg.BasePath, "traces.csv", lg.FileSuffix))
		if err != nil {
			return err
		}
		for i := range lg.currState {
			str := fmt.Sprintf("%v,%v,%d,%f", lg.currState[i], lg.prevState[i], lg.actions[i], lg.rewards[i])
			if lg.headers != nil {
				for _, val := range lg.others[i] {
					str += fmt.Sprintf(",%f", val)
				}
			}
			str += "\n"

			_, err = file.WriteString(str)
			if err != nil {
				return err
			}
		}
	}

	return nil
}
