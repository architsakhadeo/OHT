package experiment

import (
	"encoding/json"
	"errors"

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
	agent           rlglue.Agent
	environment     rlglue.Environment
	logger          rlglue.Logger
	loggerInterval  int
	numStepsTaken   int
	numEpisodesDone int
}

func New(expAttr json.RawMessage, agentAttr, envAttr rlglue.Attributes, logger rlglue.Logger) (*Experiment, error) {
	// Ensure errors are also logged
	var err error
	defer func() {
		if err != nil {
			logger.Message(err.Error())
		}
	}()

	var set settings
	err = json.Unmarshal(expAttr, &set)
	if err != nil {
		err = errors.New("Experiment settings couldn't be parsed: " + err.Error())
		return nil, err
	}

	// Check for bad settings
	if set.MaxEpisodes == nil && set.MaxSteps == nil {
		err = errors.New("Experiment settings requres either 'episodes' or 'steps'")
	} else if set.MaxEpisodes != nil && set.MaxSteps != nil {
		err = errors.New("Experiment settings requres either 'episodes' or 'steps', but not both")
	}
	if err != nil {
		return nil, err
	}

	ci := &Experiment{
		settings:       set,
		logger:         logger,
		loggerInterval: logger.Interval(),
	}

	// Set up environment
	ci.environment, err = rlglue.CreateEnvironment(set.Environment)
	if err != nil {
		return nil, err
	}
	err = ci.environment.Initialize(envAttr, logger)
	if err != nil {
		return nil, err
	}

	// Set up agent
	ci.agent, err = rlglue.CreateAgent(set.Agent)
	if err != nil {
		return nil, err
	}
	err = ci.agent.Initialize(agentAttr, ci.environment.GetAttributes(), logger)
	if err != nil {
		return nil, err
	}

	return ci, errors.New("Not implemented")
}

func (exp *Experiment) Run() {
	if exp.MaxSteps != nil {
		exp.runContinuous()
	} else {
		exp.runEpisodic()
	}

	// TODO Save the agent parameters (but for multiple runs, just do it once). They might need to be loaded from the agent in case it changed something?

	exp.logger.Save()
}

func (exp *Experiment) runContinuous() {
	exp.logger.Message("Starting continuous experiment")
	for exp.numStepsTaken < *exp.MaxSteps {
		exp.runSingleEpisode()
	}
}

func (exp *Experiment) runEpisodic() {
	exp.logger.Message("Starting episodic experiment")
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

		exp.logger.LogStep(prevState, newState, action, reward) // TODO add gamma at end

		if episodeEnded {
			exp.agent.End(newState, reward)
			// TODO this is continuous, so episodes shouldn't end. This is an error? But for cartpole, we're still using episodes sort of.
		} else {
			action = exp.agent.Step(newState, reward)
		}

		prevState = newState

		exp.numStepsTaken += 1
		numStepsThisEpisode += 1

		if exp.numStepsTaken%exp.loggerInterval == 0 {
			exp.logger.MessageDelta("total steps", exp.numStepsTaken)
		}
	}

	exp.logger.LogEpisodeLength(numStepsThisEpisode)
	if episodeEnded {
		exp.logger.MessageRewardSince(exp.numStepsTaken-numStepsThisEpisode, "episode", exp.numEpisodesDone, "total steps", exp.numStepsTaken, "episode steps", numStepsThisEpisode)
	}

	exp.numEpisodesDone += 1
}
