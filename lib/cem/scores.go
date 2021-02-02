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
	case "continuing-last-half":
		*rsg = func() RunScorer { return &scoreContinuingLastHalf{} }
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

// scoreContinuingLastHalf calculates a score in the case of continuing, where only the last half is counted
type scoreContinuingLastHalf struct {
	average float64
	numRuns int
}

func (su *scoreContinuingLastHalf) UpdateRun(listOfListOfRewards [][]float64) {
	su.numRuns++
	var listOfRewards []float64
	for i := 0; i < len(listOfListOfRewards); i++ {
		listOfRewards = append(listOfRewards, listOfListOfRewards[i]...)
	}
	fmt.Println("Length of run: ", len(listOfRewards))
	for index := int(len(listOfRewards) / 2.0); index < len(listOfRewards); index++ {
		su.average += listOfRewards[index]
	}
}

func (su *scoreContinuingLastHalf) Score() float64 {
	return su.average / float64(su.numRuns) // This division is a waste of time since numRuns is constant
}
