package experiment

import (
	"encoding/json"
	"errors"

	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

type settings struct {
	MaxEpisodes *int   `json:"episodes"`
	MaxSteps    *int   `json:"steps"`
	Agent       string `json:"agent"`
	Environment string `json:"environment"`
}

// Experiment runs an experiment.
// The experiment itself is actually just a config file which is passed in to New as json.RawMessage.
// If the JSON contains 'episodes', then the experiment runs for that many episodes. If it contains
// 'steps', it instead runs for that many steps, cutting off mid-episode if necessary.
// It currently only works in an episodic paradigm.
// The JSON must also specify 'agent' and 'environment'.
type Experiment struct {
	settings
	agent       rlglue.Agent
	environment rlglue.Environment
	logger.Debug
	logger.Data
	debugInterval   int
	numStepsTaken   int
	numEpisodesDone int
}

func New(expAttr json.RawMessage, agentAttr, envAttr rlglue.Attributes, debug logger.Debug, log logger.Data) (*Experiment, error) {
	ci := &Experiment{
		Debug:         debug,
		Data:          log,
		debugInterval: debug.Interval(),
	}

	// Ensure errors are also logged
	var err error
	defer debug.Error(&err)

	// Parse settings
	err = json.Unmarshal(expAttr, &ci.settings)
	if err != nil {
		err = errors.New("Experiment settings couldn't be parsed: " + err.Error())
		return nil, err
	}

	// Check for bad settings
	if ci.settings.MaxEpisodes == nil && ci.settings.MaxSteps == nil {
		err = errors.New("Experiment settings requres either 'episodes' or 'steps'")
	} else if ci.settings.MaxEpisodes != nil && ci.settings.MaxSteps != nil {
		err = errors.New("Experiment settings requres either 'episodes' or 'steps', but not both")
	}
	if err != nil {
		return nil, err
	}

	ci.environment, err = InitializeEnvironment(ci.settings.Environment, envAttr, debug)
	if err != nil {
		return nil, err
	}

	ci.agent, err = InitializeAgent(ci.settings.Agent, agentAttr, ci.environment, debug)
	if err != nil {
		return nil, err
	}

	return ci, nil
}

func (exp *Experiment) Run() {
	if exp.MaxSteps != nil {
		exp.runContinuous()
	} else {
		exp.runEpisodic()
	}

	// TODO Save the agent parameters (but for multiple runs, just do it once). They might need to be loaded from the agent in case it changed something?

	exp.SaveLog()
}

func (exp *Experiment) runContinuous() {
	exp.Message("msg", "Starting continuous experiment")
	for exp.numStepsTaken < *exp.MaxSteps {
		exp.runSingleEpisode()
	}
}

func (exp *Experiment) runEpisodic() {
	exp.Message("msg", "Starting episodic experiment")
	for exp.numEpisodesDone < *exp.MaxEpisodes {
		exp.runSingleEpisode()
	}
}

func (exp *Experiment) runSingleEpisode() {
	episodeEnded := false
	prevState := exp.environment.Start()
	action := exp.agent.Start(prevState)

	numStepsThisEpisode := 0
	for !episodeEnded && (exp.MaxSteps == nil || exp.numStepsTaken < *exp.MaxSteps) {
		var reward float64
		var newState rlglue.State
		newState, reward, episodeEnded = exp.environment.Step(action)

		exp.LogStep(prevState, newState, action, reward) // TODO add gamma at end

		if episodeEnded {
			exp.agent.End(newState, reward)
			// TODO this is continuous, so episodes shouldn't end. This is an error? But for cartpole, we're still using episodes sort of.
		} else {
			action = exp.agent.Step(newState, reward)
		}

		prevState = newState

		exp.numStepsTaken += 1
		numStepsThisEpisode += 1

		if exp.numStepsTaken%exp.debugInterval == 0 {
			exp.MessageDelta("total steps", exp.numStepsTaken)
		}
	}

	exp.LogEpisodeLength(numStepsThisEpisode)
	if episodeEnded {
		reward := exp.RewardSince(exp.numStepsTaken - numStepsThisEpisode)
		exp.Message("total reward", reward, "episode", exp.numEpisodesDone, "total steps", exp.numStepsTaken, "episode steps", numStepsThisEpisode)
	}

	exp.numEpisodesDone += 1
}
