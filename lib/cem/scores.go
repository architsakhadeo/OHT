package cem

import (
	"encoding/json"
	"fmt"
)

// RunScorer recevies the results of a series of runs and compiles them into a score.
type RunScorer interface {
	// UpdateRun receives the list of list of rewards from a run of the experiment.
	UpdateRun(rewards [][]float64)

	// Score returns a score for the series of runs
	Score() float64
}

type RunScorerGenerator func() RunScorer

func (rsg *RunScorerGenerator) UnmarshalJSON(data []byte) error {
	name := ""
	if err := json.Unmarshal(data, &name); err != nil {
		return err
	}

	switch name {
	case "episode-longer-is-better":
		*rsg = func() RunScorer { return &scoreEpisodeLongerIsBetter{} }
	default:
		return fmt.Errorf("couldn't find RunScorerGenerator with name '%s'", name)
	}
	return nil
}

// scoreEpisodeLongerIsBetter calculates a score in the case of episodes, where longer runs are better.
type scoreEpisodeLongerIsBetter struct {
	averageSuccess, average int
}

func (su *scoreEpisodeLongerIsBetter) UpdateRun(rewards [][]float64) {
	su.averageSuccess += len(rewards) // Add the number of successes during this run
	for i := 0; i < len(rewards); i++ {
		su.average += len(rewards[i]) // Adds the total number of steps taken in the run
	}
}

func (su *scoreEpisodeLongerIsBetter) Score() float64 {
	return -float64(su.average) / float64(su.averageSuccess) // negative of steps to failure
}
