package environment

import (
	"encoding/json"
	"fmt"

	tpo "github.com/stellentus/cartpoles/lib/util/type-opr"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

// Replay loads a previous data file and replays it.
// The previously chosen action is appended to the end of the state vector.
// When 'log-action-diff' is true, the environment logs errors if the same action was not chosen.
// The attribute "state-contains-replay" is always set to true.
type Replay struct {
	logger.Debug
	logger.ReplayData

	replaySettings

	upcomingReward float64

	previousAction rlglue.Action
	upcomingState  rlglue.State
}

type replaySettings struct {
	Path          string
	Suffix        string
	LogActionDiff bool `json:"log-action-diff"`
}

func init() {
	Add("replay", NewReplay)
}

func NewReplay(debug logger.Debug) (rlglue.Environment, error) {
	return &Replay{Debug: debug}, nil
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (env *Replay) Initialize(run uint, attr rlglue.Attributes) error {
	err := json.Unmarshal(attr, &env.replaySettings)
	if err != nil || env.replaySettings.Path == "" {
		env.Message("warning", "environment.Replay path wasn't available: "+err.Error())
	}

	env.ReplayData, err = logger.NewReplayData(env.replaySettings.Path, env.replaySettings.Suffix, env.Debug)
	if err != nil {
		return err
	}

	env.Message("replay-environment settings", fmt.Sprintf("%+v", env.replaySettings))

	return nil
}

// Start returns an initial observation.
func (env *Replay) Start(randomizeStartStateCondition bool) (rlglue.State, string) {
	st, _, _ := env.getStep()
	return st, ""
}

// Step takes an action and provides the resulting reward, the new observation, and whether the state is terminal.
// For this continuous environment, it's only terminal if the action was invalid.
func (env *Replay) Step(act rlglue.Action, randomizeStartStateCondition bool) (rlglue.State, float64, bool, string) {
	if env.LogActionDiff && act != env.previousAction {
		env.Message("warning", "offline agent got mismatched actions", "expected action", env.previousAction, "received action", act)
	}
	st, rew, stateMismatch := env.getStep()
	if stateMismatch {
		env.Message("msg", "Abrupt state transition, likely due to end of episode")
	}
	return st, rew, false, ""
}

func (env *Replay) getStep() (rlglue.State, float64, bool) {
	thisReward := env.upcomingReward
	upcomingState, currentState, expectedAction, upcomingReward, ok := env.ReplayData.NextStep()
	stateMismatch := !env.upcomingState.IsEqual(currentState)
	if ok {
		env.upcomingReward = upcomingReward
		env.previousAction = expectedAction
		env.upcomingState = upcomingState
	} // else end of file was reached, and only 'currentState was valid'
	return stateWithAction(currentState, expectedAction), thisReward, stateMismatch
}

// GetAttributes returns attributes for this environment.
func (env *Replay) GetAttributes() rlglue.Attributes {
	return rlglue.Attributes(`{"numberOfActions":4,"state-contains-replay":true}`)
	// TODO should be saved as attributes from a known struct
}

func stateWithAction(st rlglue.State, act rlglue.Action) rlglue.State {
	//return append(st, float64(act))
	actFloat, _ := tpo.GetFloat(act)
	return append(st, actFloat)
}

func (env *Replay) GetInfo(info string, value float64) interface{} {
	return nil
}