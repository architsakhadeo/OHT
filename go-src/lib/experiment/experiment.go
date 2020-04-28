package experiment

import (
	"errors"

	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

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
	numStepsTaken   int
	numEpisodesDone int
}

func New(agent rlglue.Agent, environment rlglue.Environment, set settings, debug logger.Debug, log logger.Data) (*Experiment, error) {
	ci := &Experiment{
		Debug:       debug,
		Data:        log,
		agent:       agent,
		environment: environment,
		settings:    set,
	}

	// Ensure errors are also logged
	var err error
	defer debug.Error(&err)

	// Check for bad settings
	if ci.settings.MaxEpisodes == 0 && ci.settings.MaxSteps == 0 {
		err = errors.New("Experiment settings requres either 'episodes' or 'steps'")
	} else if ci.settings.MaxEpisodes != 0 && ci.settings.MaxSteps != 0 {
		err = errors.New("Experiment settings requres either 'episodes' or 'steps', but not both")
	}
	if err != nil {
		return nil, err
	}
	return ci, nil
}

func (exp *Experiment) Run() error {
	if exp.MaxSteps != 0 {
		exp.runContinuous()
	} else {
		exp.runEpisodic()
	}

	// TODO Save the agent parameters (but for multiple runs, just do it once). They might need to be loaded from the agent in case it changed something?

	return exp.SaveLog()
}

func (exp *Experiment) runContinuous() {
	exp.Message("msg", "Starting continuous experiment")
	for exp.numStepsTaken < exp.MaxSteps {
		exp.runSingleEpisode()
	}
}

func (exp *Experiment) runEpisodic() {
	exp.Message("msg", "Starting episodic experiment")
	for exp.numEpisodesDone < exp.MaxEpisodes {
		exp.runSingleEpisode()
	}
}

func (exp *Experiment) runSingleEpisode() {
	episodeEnded := false
	prevState := exp.environment.Start()
	action := exp.agent.Start(prevState)

	numStepsThisEpisode := 0
	for !episodeEnded && (exp.MaxSteps == 0 || exp.numStepsTaken < exp.MaxSteps) {
		var reward float64
		var newState rlglue.State
		newState, reward, episodeEnded = exp.environment.Step(action)

		exp.LogStep(prevState, newState, action, reward) // TODO add gamma at end

		if episodeEnded {
			if exp.MaxSteps != 0 {
				exp.Message("warning", "An episode ended in a continuing setting. This doesn't make sense.")
			}
			exp.agent.End(newState, reward)
		} else {
			action = exp.agent.Step(newState, reward)
		}

		prevState = newState

		exp.numStepsTaken += 1
		numStepsThisEpisode += 1

		if exp.numStepsTaken%exp.DebugInterval == 0 {
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
