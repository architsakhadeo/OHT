package agent

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

type handControllerSettings struct {
	EnableDebug bool `json:"enable-debug"`

	// PlanDuration is the number of actions that should be taken before looking at state again, minimum 1.
	// 	- 2 gives optimal behavior, insensitive to 'Threshold'.
	// 	- 0.2s is an appropriate human reaction time, which could also be used as a the time it takes for a human to
	// 	- change plans, even though that's not necessarily the same number.)
	// 	- This would correspond to a plan duration of reaction_time/tau = 0.2/0.02 = 10.
	PlanDuration int `json:"plan-duration"`

	// Threshold is a parameter between 0 and 1 to control behavior.
	Threshold float64 `json:"Threshold"`

	// FailAngle is the angle at which the environment terminates the episode.
	// It's imported in degrees but converted to radians.
	FailAngle float64 `json:"fail-degrees"`

	// FailPosition is the position at which the environment terminates the episode, whether it's positive or negative.
	FailPosition float64 `json:"fail-position"`
}

type HandController struct {
	logger.Debug
	handControllerSettings

	actions     []int
	actionIndex int
}

func init() {
	Add("hand-controller", NewHandController)
}

func NewHandController(logger logger.Debug) (rlglue.Agent, error) {
	return &HandController{Debug: logger}, nil
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *HandController) Initialize(run uint, expAttr, envAttr rlglue.Attributes) error {
	// Set defaults, which will be overridden if the JSON contains different values.
	agent.handControllerSettings = handControllerSettings{
		PlanDuration: 2,
		Threshold:    0.9,
		FailAngle:    15,
		FailPosition: 2.4,
	}

	err := json.Unmarshal(expAttr, &agent.handControllerSettings)
	if err != nil {
		agent.Message("warning", "agent.HandController settings weren't available: "+err.Error())
	}

	if agent.EnableDebug {
		agent.Message("msg", "agent.HandController Initialize",
			"plan-duration", agent.PlanDuration,
			"Threshold", agent.Threshold,
			"fail-degrees", agent.FailAngle,
			"fail-position", agent.FailPosition,
		)
	}

	agent.FailAngle /= 180 * math.Pi // convert degrees to radians

	agent.actions = make([]int, agent.PlanDuration)

	agent.Message("hand-controller settings", fmt.Sprintf("%+v", agent.handControllerSettings))

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *HandController) Start(state rlglue.State) rlglue.Action {
	if agent.EnableDebug {
		agent.Message("msg", "episode start")
	}

	agent.actions = make([]int, agent.PlanDuration)

	return agent.chooseAction(state)
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *HandController) Step(state rlglue.State, reward float64) rlglue.Action {
	return agent.chooseAction(state)
}

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *HandController) End(state rlglue.State, reward float64) {
	if agent.EnableDebug {
		agent.Message("msg", "end", "state", state, "reward", reward)
	}
}

// chooseAction allows the controller to act on a much slower timestep. (During each time period,
// it can act with a specified ratio of left/right actions.) It also tiles into large tiles. Based
// on the current tile, choose a pre-set action or action series and follow it for a while.
// Then see which tile I'm in and make a new choice.
func (agent *HandController) chooseAction(state rlglue.State) rlglue.Action {
	if agent.actionIndex >= agent.PlanDuration {
		agent.selectActions(state)
	}

	action := agent.actions[agent.actionIndex]
	agent.actionIndex++
	return rlglue.Action(action)
}

// createActionSeries creates a predetermined series of actions for the next actions_per_step steps.
// `level` should be a number between 0 and 1. It's the average action value for this time period.
func (agent *HandController) createActionSeries(level float64) {
	// We expect after `actions_per_step` steps, the sum of actions should be `level*actions_per_step`.
	// So at each step, we decide which action will keep the average level closest to `level`.
	sm := 0
	for i := range agent.actions {
		target_sum := level * float64(i) // By this time, the sum should be as close as possible to this value.

		// If the current sum is within 0.5 of the target, action is 0. Otherwise, the sum is too low and we need to increase it.

		next_action := 0
		if float64(sm)+0.5 < target_sum {
			next_action = 1
		}

		sm += next_action
		agent.actions[i] = next_action
	}
	agent.actionIndex = 0
}

// Same as createActionSeries, but the input ranges from -1 to 1.
func (agent *HandController) scaledCreateActionSeries(scaled_level float64) {
	agent.createActionSeries((scaled_level + 1) / 2)
}

// selectActions chooses the next action series based on the current state.
func (agent *HandController) selectActions(state rlglue.State) {
	angle := state[2]

	// This code will try to keep the angle balanced, but ignores the position condition.
	// I think it still usually fails to keep the pole up for more than 2–3s.

	// Respond in proportion to how far we've tilted
	if math.Abs(angle) > agent.Threshold*agent.FailAngle {
		// Just do a maximum movement in the same direction
		agent.scaledCreateActionSeries(angle / math.Abs(angle))
	} else {
		// Just do a proportional movement in the same direction
		agent.scaledCreateActionSeries(angle / agent.FailAngle)
	}
}

func (agent *HandController) GetLock() bool {
	return false
}

func (agent *HandController) SaveWeights(basePath string) error {
	return nil
}

func (agent *HandController) GetLearnProg() float64 {
	return float64(0)
}
