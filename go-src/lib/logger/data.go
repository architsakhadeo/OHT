package logger

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path"
	"strconv"
	"strings"

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

	file, err := os.Create(path.Join(lg.BasePath, "rewards-"+lg.FileSuffix+".csv"))
	if err != nil {
		return err
	}
	// Write header row
	_, err = file.WriteString("rewards\n")
	if err != nil {
		return err
	}
	// Write remaining rows
	for _, rew := range lg.rewards {
		_, err = file.WriteString(fmt.Sprintf("%f\n", rew))
		if err != nil {
			return err
		}
	}

	if lg.ShouldLogEpisodeLengths {
		file, err := os.Create(path.Join(lg.BasePath, "episodes-"+lg.FileSuffix+".csv"))
		if err != nil {
			return err
		}
		// Write header row
		_, err = file.WriteString("episode lengths\n")
		if err != nil {
			return err
		}
		// Write remaining rows
		for _, ep := range lg.episodeLengths {
			_, err = file.WriteString(fmt.Sprintf("%d\n", ep))
			if err != nil {
				return err
			}
		}
	}

	if lg.ShouldLogTraces {
		file, err := os.Create(path.Join(lg.BasePath, "traces-"+lg.FileSuffix+".csv"))
		if err != nil {
			return err
		}

		// Write header row
		str := "new state,previous state,action,reward"
		if len(lg.headers) > 0 {
			for _, hdr := range lg.headers {
				str += "," + hdr
			}
		}
		str += "\n"
		_, err = file.WriteString(str)
		if err != nil {
			return err
		}

		// Write remaining rows
		for i := range lg.currState {
			str := fmt.Sprintf("%v,%v,%d,%f", lg.currState[i], lg.prevState[i], lg.actions[i], lg.rewards[i])
			if len(lg.headers) > 0 {
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

func (lg *dataLogger) loadLog(pth string, suffix string, loadRewards, loadEpisodes, loadTraces bool) error {
	lg.DataConfig = DataConfig{
		ShouldLogTraces:         loadTraces,
		ShouldLogEpisodeLengths: loadEpisodes,
		BasePath:                pth,
		FileSuffix:              suffix,
	}
	lg.episodeLengths = []int{}
	lg.prevState = []rlglue.State{}
	lg.currState = []rlglue.State{}
	lg.actions = []rlglue.Action{}
	lg.rewards = []float64{}
	lg.others = [][]float64{}

	if loadRewards && !loadTraces { // If traces exists, don't bother with rewards
		file, err := os.Open(path.Join(lg.BasePath, "rewards-"+lg.FileSuffix+".csv"))
		if err != nil {
			return err
		}
		defer file.Close()

		scanner := bufio.NewScanner(file)
		if !scanner.Scan() {
			return errors.New("Reward file was empth at '" + lg.BasePath + "'")
		} // else assume header is correct

		for scanner.Scan() {
			var val float64
			_, err = fmt.Sscanf(scanner.Text(), "%f", &val)
			if err != nil {
				return err
			}
			lg.rewards = append(lg.rewards, val)
		}

		if err := scanner.Err(); err != nil {
			return err
		}
	}

	if loadEpisodes {
		file, err := os.Open(path.Join(lg.BasePath, "episodes-"+lg.FileSuffix+".csv"))
		if err != nil {
			return err
		}
		defer file.Close()

		scanner := bufio.NewScanner(file)
		if !scanner.Scan() {
			return errors.New("Reward file was empth at '" + lg.BasePath + "'")
		} // else assume header is correct

		for scanner.Scan() {
			var val int
			_, err = fmt.Sscanf(scanner.Text(), "%d", &val)
			if err != nil {
				return err
			}
			lg.episodeLengths = append(lg.episodeLengths, val)
		}

		if err := scanner.Err(); err != nil {
			return err
		}
	}

	if loadTraces {
		file, err := os.Open(path.Join(lg.BasePath, "traces.csv-"+lg.FileSuffix+".csv"))
		if err != nil {
			return err
		}
		defer file.Close()

		scanner := bufio.NewScanner(file)
		if !scanner.Scan() {
			return errors.New("Reward file was empth at '" + lg.BasePath + "'")
		}
		headers := strings.Split(scanner.Text(), ",")
		if len(headers) > 4 {
			lg.headers = []string{}
			for i := 4; i < len(headers); i++ {
				lg.headers = append(lg.headers, headers[i])
			}
		}

		for scanner.Scan() {
			values := strings.Split(scanner.Text(), ",")
			if val, err := parseState(values[0]); err != nil {
				return err
			} else {
				lg.currState = append(lg.currState, val)
			}
			if val, err := parseState(values[1]); err != nil {
				return err
			} else {
				lg.prevState = append(lg.prevState, val)
			}
			lg.actions = append(lg.actions, parseActionDefaultZero(values[2]))
			lg.rewards = append(lg.rewards, parseFloatDefaultZero(values[3]))
			if len(values) > 4 {
				others := []float64{}
				for i := 4; i < len(values); i++ {
					others = append(others, parseFloatDefaultZero(values[i]))
				}
				lg.others = append(lg.others, others)
			}
		}

		if err := scanner.Err(); err != nil {
			return err
		}

		numSteps := len(lg.rewards)
		if numSteps != len(lg.prevState) || numSteps != len(lg.currState) || numSteps != len(lg.actions) {
			return fmt.Errorf("Data file CSV unequal columns: %d %d %d %d", len(lg.currState), len(lg.prevState), len(lg.actions), numSteps)
		}
		for i := range lg.headers {
			if len(lg.others[i]) != numSteps {
				return fmt.Errorf("Data file CSV extra column %d has %d rows instead of %d", i, len(lg.others[i]), numSteps)
			}
		}
	}

	return nil
}

func parseState(str string) (rlglue.State, error) {
	str = strings.TrimSpace(str)
	if str[0] != '[' && str[len(str)-1] != ']' {
		return nil, errors.New("Could not parse state '" + str + "'")
	}
	state := rlglue.State{}
	values := strings.Split(str[1:len(str)-1], ",")
	for _, strVal := range values {
		if val, err := strconv.ParseFloat(strVal, 64); err != nil {
			fmt.Println("values", values)
			return nil, errors.New("Could not parse state value '" + strVal + "' from '" + str + "': " + err.Error())
		} else {
			state = append(state, val)
		}
	}
	return state, nil
}

func parseActionDefaultZero(str string) rlglue.Action {
	if val, err := strconv.Atoi(str); err != nil {
		return 0
	} else {
		return rlglue.Action(val)
	}
}

func parseFloatDefaultZero(str string) float64 {
	if val, err := strconv.ParseFloat(str, 64); err != nil {
		return 0
	} else {
		return val
	}
}
