package logger

import (
	"fmt"

	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
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

func (rd ReplayData) NextStep() (rlglue.State, rlglue.State, rlglue.Action, float64) {
	if rd.nextStep >= len(rd.rewards) {
		rd.Message("err", fmt.Sprintf("Attempted to replay step beyond maximum of %d", len(rd.rewards)))
		rd.Reset() // This could result in an infinite loop if no data was loaded.
	}

	a, b, c, d := rd.currState[rd.nextStep], rd.prevState[rd.nextStep], rd.actions[rd.nextStep], rd.rewards[rd.nextStep]
	rd.nextStep++
	return a, b, c, d
}

func (rd ReplayData) PeekNextCurrentState() rlglue.State {
	if rd.nextStep >= len(rd.rewards) {
		rd.Message("err", fmt.Sprintf("Attempted to peek at step beyond maximum of %d", len(rd.rewards)))
		return nil
	}
	return rd.currState[rd.nextStep]
}

func (rd *ReplayData) Reset() {
	rd.nextStep = 0
}
