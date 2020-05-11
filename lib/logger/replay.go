package logger

import (
	"fmt"

	"github.com/stellentus/cartpoles/lib/rlglue"
)

// ReplayData is used to replay Data.
// TODO consider whether Data stores and replays episodes correctly, since terminal states are visited.
type ReplayData struct {
	dataLogger
	numSteps int
	nextStep int
}

func NewReplayData(pth, suffix string, debug Debug) (ReplayData, error) {
	rd := ReplayData{
		dataLogger: dataLogger{
			Debug: debug,
		},
	}
	err := rd.dataLogger.loadLog(pth, suffix, false, false, true)
	return rd, err
}

func (rd ReplayData) Start() rlglue.State {
	if rd.nextStep != 0 {
		rd.Message("err", fmt.Sprintf("Attempted to replay start while at step %d", rd.nextStep))
	}
	return rd.prevState[0]
}

// The final argument is false if the end of the file was reached
func (rd *ReplayData) NextStep() (rlglue.State, rlglue.State, rlglue.Action, float64, bool) {
	if rd.nextStep >= len(rd.rewards) {
		if rd.nextStep == len(rd.rewards) {
			rd.nextStep++
			return nil, rd.currState[len(rd.rewards)-1], 0, 0, false // At least the previous state can be provided
		} else {
			rd.Message("err", fmt.Sprintf("Attempted to replay step beyond maximum of %d", len(rd.rewards)))
			rd.Reset() // This could result in an infinite loop if no data was loaded.
		}
	}

	a, b, c, d := rd.currState[rd.nextStep], rd.prevState[rd.nextStep], rd.actions[rd.nextStep], rd.rewards[rd.nextStep]
	rd.nextStep++
	return a, b, c, d, true
}

func (rd *ReplayData) Reset() {
	rd.nextStep = 0
}
