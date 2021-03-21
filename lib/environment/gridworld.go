package environment

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

const (
	gridworldWidth = 4.0
)

var (
	startPos = []float64{0.0, 0.0}
	goalPos  = []float64{3.0, 3.0}
)

type GridworldSettings struct {
	Seed              int64     `json:"seed"`
	Delays            []int     `json:"delays"`
	PercentNoiseState []float64 `json:"percent_noise"`
}

type Gridworld struct {
	logger.Debug
	GridworldSettings
	state             rlglue.State
	rng               *rand.Rand
	buffer            [][]float64
	bufferInsertIndex []int
	actions           [][]float64
}

func init() {
	Add("gridworld", NewGridworld)
}

// TODO debug
func NewGridworld(logger logger.Debug) (rlglue.Environment, error) {
	return &Gridworld{Debug: logger}, nil
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (env *Gridworld) Initialize(run uint, attr rlglue.Attributes) error {
	set := GridworldSettings{}
	err := json.Unmarshal(attr, &set)
	if err != nil {
		err = errors.New("environment.Gridworld settings error: " + err.Error())
		env.Message("err", err)
		return err
	}
	set.Seed += int64(run)

	return env.InitializeWithSettings(set)
}

func (env *Gridworld) InitializeWithSettings(set GridworldSettings) error {
	env.GridworldSettings = set
	//fmt.Println("Env Seed: ", env.Seed)
	//fmt.Println("Env GridworldSettings Seed: ", env.GridworldSettings.Seed)
	//fmt.Println("Set Seed:", set.Seed)
	//fmt.Println("Seed actually used by the environment: ", env.Seed)
	env.rng = rand.New(rand.NewSource(env.Seed)) // Create a new rand source for reproducibility

	// actions
	// 0.0 = left
	// 1.0 = right
	// 2.0 = up
	// 3.0 = down
	//env.actions = []float64{0.0, 1.0, 2.0, 3.0}

	if len(env.PercentNoiseState) == 1 {
		// Copy it for all dimensions
		noise := env.PercentNoiseState[0]
		env.PercentNoiseState = []float64{noise, noise}
	} else if len(env.PercentNoiseState) != 2 && len(env.PercentNoiseState) != 0 {
		err := fmt.Errorf("environment.Puddleworld requires percent_noise to be length 2, 1, or 0, not length %d", len(env.PercentNoiseState))
		env.Message("err", err)
		return err
	}

	if len(env.Delays) == 1 {
		// Copy it for all dimensions
		delay := env.Delays[0]
		env.Delays = []int{delay, delay}
	} else if len(env.Delays) != 2 && len(env.Delays) != 0 {
		err := fmt.Errorf("environment.Puddleworld requires delays to be length 2, 1, or 0, not length %d", len(env.Delays))
		env.Message("err", err)
		return err
	}

	// If noise is off, set array to nil
	totalNoise := 0.0
	for _, noise := range env.PercentNoiseState {
		totalNoise += noise
	}
	if totalNoise == 0.0 {
		env.PercentNoiseState = nil
	}

	// If delays are off, set array to nil
	totalDelay := 0
	for _, noise := range env.Delays {
		totalDelay += noise
	}
	if totalDelay == 0 {
		env.Delays = nil
	}

	env.state = make(rlglue.State, 2)

	if len(env.Delays) != 0 {
		env.buffer = make([][]float64, 2)
		for i := range env.buffer {
			env.buffer[i] = make([]float64, env.Delays[i])
		}
		env.bufferInsertIndex = make([]int, 2)
	}

	env.Message("grid world settings", fmt.Sprintf("%+v", env.GridworldSettings))

	return nil
}

func (env *Gridworld) noisyState() rlglue.State {
	stateLowerBound := []float64{0.0, 0.0}
	stateUpperBound := []float64{1.0, 1.0}

	state := make(rlglue.State, 2)
	copy(state, env.state)

	if len(env.PercentNoiseState) != 0 {
		// Only add noise if it's configured
		for i := range state {
			state[i] += env.randFloat(env.PercentNoiseState[i]*stateLowerBound[i], env.PercentNoiseState[i]*stateUpperBound[i])
			state[i] = env.clamp(state[i], stateLowerBound[i], stateUpperBound[i])
		}
	}
	return state
}

func (env *Gridworld) clamp(x float64, min float64, max float64) float64 {
	return math.Max(min, math.Min(x, max))
}

func (env *Gridworld) randomizeState(randomizeStartStateCondition bool) {
	if randomizeStartStateCondition == false {
		copy(env.state, startState)
		return
	}

	for i := range env.state {
		env.state[i] = env.randInt(0.0, 3.0)
	}
	/*
		for true {
			for i := range env.state {
				env.state[i] = env.randInt(0.0, 3.0)
			}
			if env.state[0] == 3.0 && env.state[1] == 3.0 {
				continue
			} else {
				break
			}
		}
	*/
}

func (env *Gridworld) randFloat(min, max float64) float64 {
	return env.rng.Float64()*(max-min) + min
}

func (env *Gridworld) randInt(min, max float64) float64 {
	float := env.rng.Float64()
	diff := 0.25 //1.0/(max - min + 1.0)
	returnvalue := 0.0
	if float < diff {
		returnvalue = 0.0
	} else if float >= diff && float < 2.0*diff {
		returnvalue = 1.0
	} else if float >= 2.0*diff && float < 3.0*diff {
		returnvalue = 2.0
	} else if float >= 3.0*diff && float < 4.0*diff {
		returnvalue = 3.0
	}
	return returnvalue
}

// func (env *Puddleworld) gaussian1d(p, mu, sig float64) float64 {
// 	return math.Exp(-math.Pow(p-mu, 2)/(2.*math.Pow(sig, 2))) / (sig * math.Sqrt(2.*math.Pi))
// }

// Start returns an initial observation.
func (env *Gridworld) Start(randomizeStartStateCondition bool) (rlglue.State, string) {
	env.randomizeState(randomizeStartStateCondition)
	//fmt.Println("Start state", env.state)
	return env.getObservations(), ""
}

// Step takes an action and provides the resulting reward, the new observation, and whether the state is terminal.
// For this continuous environment, it's only terminal if the action was invalid.
func (env *Gridworld) Step(act rlglue.Action, randomizeStartStateCondition bool) (rlglue.State, float64, bool, string) {
	//fmt.Println("Current state: ", env.state)
	//fmt.Println("Action: ", act)
	reward := env.getRewards(env.state, act)
	//fmt.Println("Reward: ", reward)
	//for i := range env.state {
	//	env.state[i] += env.actions[act.(int)][i] + env.rng.NormFloat64()*0.01
	//	env.state[i] = env.clamp(env.state[i], 0.0, 1.0)
	//}

	env.state = env.getNextState(env.state, act)
	//fmt.Println("Next state: ", env.state)

	obs := env.getObservations()
	var done bool
	if obs[0] == goalPos[0] && obs[1] == goalPos[1] {
		done = true
	} else {
		done = false
	}
	//fmt.Println("Terminal: ", done)
	//fmt.Println("")
	//fmt.Println("")

	return obs, reward, done, ""
}

func (env *Gridworld) getObservations() rlglue.State {
	// Add noise to state to get observations
	observations := env.noisyState()

	// Add delays
	if len(env.Delays) != 0 {
		for i, obs := range observations {
			observations[i] = env.buffer[i][env.bufferInsertIndex[i]]                 // Load the delayed observation
			env.buffer[i][env.bufferInsertIndex[i]] = obs                             // Store the current state
			env.bufferInsertIndex[i] = (env.bufferInsertIndex[i] + 1) % env.Delays[i] // Update the insertion point
		}
	}
	return observations
}

func (env *Gridworld) getNextState(state rlglue.State, act rlglue.Action) rlglue.State {
	// actions
	// 0.0 = left
	// 1.0 = right
	// 2.0 = up
	// 3.0 = down

	s0, s1 := state[0], state[1]
	a := act.(int)

	if a == 0.0 {
		s0 -= 1.0
	} else if a == 1.0 {
		s0 += 1.0
	} else if a == 2.0 {
		s1 += 1.0
	} else if a == 3.0 {
		s1 -= 1.0
	}

	if s0 < 0.0 {
		s0 = 0.0
	} else if s0 > 3.0 {
		s0 = 3.0
	}

	if s1 < 0.0 {
		s1 = 0.0
	} else if s1 > 3.0 {
		s1 = 3.0
	}

	nextstate := make(rlglue.State, 2)
	nextstate[0] = s0
	nextstate[1] = s1

	return nextstate
}

func (env *Gridworld) getRewards(state rlglue.State, act rlglue.Action) float64 {
	reward := -1.0

	s0, s1 := state[0], state[1]
	a := act.(int)

	// actions
	// 0.0 = left
	// 1.0 = right
	// 2.0 = up
	// 3.0 = down

	if s0 == 0.0 && s1 == 0.0 && a == 1.0 {
		reward = 0.0
	} else if s0 == 1.0 && s1 == 0.0 && a == 2.0 {
		reward = 0.0
	} else if s0 == 1.0 && s1 == 1.0 && a == 1.0 {
		reward = 0.0
	} else if s0 == 2.0 && s1 == 1.0 && a == 3.0 {
		reward = 0.0
	} else if s0 == 2.0 && s1 == 0.0 && a == 1.0 {
		reward = 0.0
	} else if s0 == 3.0 && s1 == 0.0 && a == 2.0 {
		reward = 0.0
	} else if s0 == 3.0 && s1 == 1.0 && a == 2.0 {
		reward = 0.0
	} else if s0 == 3.0 && s1 == 2.0 && a == 0.0 {
		reward = 0.0
	} else if s0 == 2.0 && s1 == 2.0 && a == 0.0 {
		reward = 0.0
	} else if s0 == 1.0 && s1 == 2.0 && a == 2.0 {
		reward = 0.0
	} else if s0 == 1.0 && s1 == 3.0 && a == 1.0 {
		reward = 0.0
	} else if s0 == 2.0 && s1 == 3.0 && a == 1.0 {
		reward = 0.0
	}

	return reward
}

func (env *Gridworld) inRange(x, min, max float64) bool {
	return x >= min && x <= max
}

// GetAttributes returns attributes for this environment.
func (env *Gridworld) GetAttributes() rlglue.Attributes {
	// Add elements to attributes.
	attributes := struct {
		NumAction  int       `json:"numberOfActions"`
		StateDim   int       `json:"stateDimension"`
		StateRange []float64 `json:"stateRange"`
	}{
		4, // 5
		2,
		[]float64{1.0, 1.0},
	}

	attr, err := json.Marshal(&attributes)
	if err != nil {
		env.Message("err", "environment.Puddleworld could not Marshal its JSON attributes: "+err.Error())
	}

	return attr
}
