package experiment

import (
	"errors"

	"github.com/stellentus/cartpoles/lib/config"
	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

// Experiment runs an experiment.
// The experiment itself is actually just a config file which is passed in to New as json.RawMessage.
// If the JSON contains 'episodes', then the experiment runs for that many episodes. If it contains
// 'steps', it instead runs for that many steps, cutting off mid-episode if necessary.
// It currently only works in an episodic paradigm.
// The JSON must also specify 'agent' and 'environment'.
type Experiment struct {
	Settings    config.Experiment
	agent       rlglue.Agent
	environment rlglue.Environment
	logger.Debug
	logger.Data
	numStepsTaken   int
	numEpisodesDone int
}

func New(agent rlglue.Agent, environment rlglue.Environment, set config.Experiment, debug logger.Debug, log logger.Data) (*Experiment, error) {
	ci := &Experiment{
		Debug:       debug,
		Data:        log,
		agent:       agent,
		environment: environment,
		Settings:    set,
	}

	// Ensure errors are also logged
	var err error
	defer debug.Error(&err)

	// Check for bad settings
	if ci.Settings.MaxEpisodes == 0 && ci.Settings.MaxSteps == 0 {
		err = errors.New("Experiment settings requres either 'episodes' or 'steps'")
	} else if ci.Settings.MaxEpisodes != 0 && ci.Settings.MaxSteps != 0 {
		err = errors.New("Experiment settings requres either 'episodes' or 'steps', but not both")
	}
	if err != nil {
		return nil, err
	}
	return ci, nil
}

func (exp *Experiment) Run() error {
	if exp.Settings.MaxSteps != 0 {
		exp.runContinuous()
	} else {
		exp.runEpisodic()
	}

	// TODO Save the agent parameters (but for multiple runs, just do it once). They might need to be loaded from the agent in case it changed something?

	return exp.SaveLog()
}

func (exp *Experiment) runContinuous() {
	exp.Message("msg", "Starting continuous experiment")
	exp.runSingleEpisode()
}

func (exp *Experiment) runEpisodic() {
	exp.Message("msg", "Starting episodic experiment")
	for exp.numEpisodesDone < exp.Settings.MaxEpisodes {
		exp.runSingleEpisode()
	}
}

func (exp *Experiment) runSingleEpisode() {
	prevState := exp.environment.Start()
	action := exp.agent.Start(prevState)

	numStepsThisEpisode := 0
	for exp.Settings.MaxSteps == 0 || exp.numStepsTaken < exp.Settings.MaxSteps {
		newState, reward, episodeEnded := exp.environment.Step(action)

		exp.LogStep(prevState, newState, action, reward) // TODO add gamma at end
		prevState = newState

		exp.numStepsTaken += 1
		numStepsThisEpisode += 1

		if exp.Settings.DebugInterval != 0 && exp.numStepsTaken%exp.Settings.DebugInterval == 0 {
			exp.MessageDelta("total steps", exp.numStepsTaken)
		}

		if !episodeEnded {
			action = exp.agent.Step(newState, reward)
			continue
		} else if exp.Settings.MaxSteps != 0 {
			// An episodic environment is being treated as continuous, so reset the environment
			newState = exp.environment.Start()
			episodeEnded = false
			action = exp.agent.Step(newState, reward)
		}

		exp.logEndOfEpisode(numStepsThisEpisode)
		exp.numEpisodesDone += 1
		numStepsThisEpisode = 0

		if exp.Settings.MaxSteps == 0 {
			// We're in the episodic setting, so we are done with this episode
			exp.agent.End(newState, reward)
			break
		}
	}

	if numStepsThisEpisode > 0 || exp.Settings.MaxSteps == 0 {
		// If there are leftover steps, we're ending after a partial episode.
		// If there aren't leftover steps, but we're in the continuing setting, this adds a '0' to indicate the previous episode terminated on a failure.
		exp.logEndOfEpisode(numStepsThisEpisode)
	}
}

func (exp *Experiment) logEndOfEpisode(numStepsThisEpisode int) {
	exp.LogEpisodeLength(numStepsThisEpisode)
	reward := exp.RewardSince(exp.numStepsTaken - numStepsThisEpisode)
	if exp.Settings.DebugInterval != 0 {
		exp.Message("total reward", reward, "episode", exp.numEpisodesDone, "total steps", exp.numStepsTaken, "episode steps", numStepsThisEpisode)
	}
}
