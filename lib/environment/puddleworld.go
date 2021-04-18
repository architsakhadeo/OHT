package environment

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	"gonum.org/v1/gonum/floats"
)

// Puddle world from http://incompleteideas.net/papers/sutton-96.pdf
const (
	goalThreshold = 0.1
	actionThrust  = 0.05
	puddleWidth   = 0.1
)

var (
	startState    = []float64{0.0, 0.0}
	goalState     = []float64{1.0, 1.0}
	puddleCenters = [][]float64{{0.1, 0.75}, {0.45, 0.75}, {0.45, 0.4}, {0.45, 0.8}}
)

type PuddleworldSettings struct {
	Seed              int64     `json:"seed"`
	Delays            []int     `json:"delays"`
	PercentNoiseState []float64 `json:"percent_noise"`
	StartHard         bool      `json:"start-hard"`
}

type Puddleworld struct {
	logger.Debug
	PuddleworldSettings
	state             rlglue.State
	rng               *rand.Rand
	buffer            [][]float64
	bufferInsertIndex []int
	actions           [][]float64
}

func init() {
	Add("puddleworld", NewPuddleworld)
}

// TODO debug
func NewPuddleworld(logger logger.Debug) (rlglue.Environment, error) {
	return &Puddleworld{Debug: logger}, nil
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (env *Puddleworld) Initialize(run uint, attr rlglue.Attributes) error {
	set := PuddleworldSettings{}
	err := json.Unmarshal(attr, &set)
	if err != nil {
		err = errors.New("environment.Puddleworld settings error: " + err.Error())
		env.Message("err", err)
		return err
	}
	set.Seed += int64(run)

	return env.InitializeWithSettings(set)
}

func (env *Puddleworld) InitializeWithSettings(set PuddleworldSettings) error {
	env.PuddleworldSettings = set
	//fmt.Println("Env Seed: ", env.Seed)
	//fmt.Println("Env PuddleworldSettings Seed: ", env.PuddleworldSettings.Seed)
	//fmt.Println("Set Seed:", set.Seed)
	//fmt.Println("Seed actually used by the environment: ", env.Seed)
	env.rng = rand.New(rand.NewSource(env.Seed)) // Create a new rand source for reproducibility

	env.actions = make([][]float64, 4) // 5
	for i := range env.actions {
		env.actions[i] = make([]float64, 2)
		// if i >= 4 {
		// 	break
		// }
		env.actions[i][i/2] = actionThrust * float64(i%2*2-1)
	}

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

	env.Message("puddle world settings", fmt.Sprintf("%+v", env.PuddleworldSettings))

	return nil
}

func (env *Puddleworld) noisyState() rlglue.State {
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

func (env *Puddleworld) clamp(x float64, min float64, max float64) float64 {
	return math.Max(min, math.Min(x, max))
}

func (env *Puddleworld) randomizeState(randomizeStartStateCondition bool) {
	if env.StartHard {
		rnd_x := env.randFloat(0.3, 0.35)
		rnd_y := env.randFloat(0.6, 0.65)
		startState[0] = rnd_x
		startState[1] = rnd_y
	}
	if randomizeStartStateCondition == false {
		copy(env.state, startState)
		return
	}

	for true {
		for i := range env.state {
			env.state[i] = env.randFloat(0.0, 1.0)
		}
		// randomly start in non-goal region states
		if floats.Distance(env.state, goalState, 1) >= goalThreshold {
			return
		}

	}
}

func (env *Puddleworld) randFloat(min, max float64) float64 {
	return env.rng.Float64()*(max-min) + min
}

// func (env *Puddleworld) gaussian1d(p, mu, sig float64) float64 {
// 	return math.Exp(-math.Pow(p-mu, 2)/(2.*math.Pow(sig, 2))) / (sig * math.Sqrt(2.*math.Pi))
// }

// Start returns an initial observation.
func (env *Puddleworld) Start(randomizeStartStateCondition bool) (rlglue.State, string) {
	env.randomizeState(randomizeStartStateCondition)
	//fmt.Println("Random start state: ", env.state)
	return env.getObservations(), ""
}

// Step takes an action and provides the resulting reward, the new observation, and whether the state is terminal.
// For this continuous environment, it's only terminal if the action was invalid.
func (env *Puddleworld) Step(act rlglue.Action, randomizeStartStateCondition bool) (rlglue.State, float64, bool, string) {
	for i := range env.state {
		env.state[i] += env.actions[act.(int)][i] + env.rng.NormFloat64()*0.01
		env.state[i] = env.clamp(env.state[i], 0.0, 1.0)
	}

	obs := env.getObservations()
	reward := env.getRewards()
	done := floats.Distance(env.state, goalState, 1) < goalThreshold

	return obs, reward, done, ""
}

func (env *Puddleworld) getObservations() rlglue.State {
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

func (env *Puddleworld) getRewards() float64 {
	reward := -1.

	dist := 0.0
	if env.state[0] < puddleCenters[0][0] && floats.Distance(env.state, puddleCenters[0], 2) < puddleWidth {
		// Left semicircle
		dist = puddleWidth - floats.Distance(env.state, puddleCenters[0], 2)
	} else if env.inRange(env.state[0], puddleCenters[0][0], puddleCenters[1][0]-puddleWidth) &&
		env.inRange(env.state[1], puddleCenters[0][1]-puddleWidth, puddleCenters[0][1]+puddleWidth) {
		// Left rectangle
		dist = puddleWidth - math.Abs(env.state[1]-puddleCenters[0][1])
	} else if env.state[1] > puddleCenters[1][1]+puddleWidth && floats.Distance(env.state, puddleCenters[3], 2) < puddleWidth {
		// Top right semicircle
		dist = puddleWidth - floats.Distance(env.state, puddleCenters[3], 2)
	} else if env.state[1] < puddleCenters[2][1] && floats.Distance(env.state, puddleCenters[2], 2) < puddleWidth {
		// Bottom right semicircle
		dist = puddleWidth - floats.Distance(env.state, puddleCenters[2], 2)
	} else if env.inRange(env.state[0], puddleCenters[2][0]-puddleWidth, puddleCenters[2][0]+puddleWidth) {
		if env.inRange(env.state[1], puddleCenters[2][1], puddleCenters[1][1]-puddleWidth) {
			// Bottom right rectangle
			dist = puddleWidth - math.Abs(env.state[0]-puddleCenters[2][0])
		} else if env.inRange(env.state[1], puddleCenters[1][1]-puddleWidth, puddleCenters[1][1]+puddleWidth) {
			// Top right "rectangle"
			if env.state[1] >= puddleCenters[3][1] {
				// The dist is either to the arc or the short edge
				if env.state[0] < (9.0-math.Sqrt(3))/20.0 {
					// Small rectangle
					dist = puddleCenters[1][1] + puddleWidth - env.state[1]
				} else if env.state[0]*0.57735+env.state[1] < 1.05981 {
					dist = floats.Distance(env.state, []float64{(9.0 - math.Sqrt(3)) / 20.0, puddleCenters[1][1] + puddleWidth}, 2)
				} else {
					dist = puddleWidth - floats.Distance(env.state, puddleCenters[3], 2)
				}
			} else {
				if env.state[0] < (9.0-math.Sqrt(3))/20.0 {
					distToBottomLeftCorner := floats.Distance(env.state,
						[]float64{puddleCenters[1][0] - puddleWidth, puddleCenters[1][1] - puddleWidth}, 2)
					distToTopEdge := puddleCenters[1][1] + puddleWidth - env.state[1]
					dist = math.Min(distToBottomLeftCorner, distToTopEdge)
				} else {
					distToTopLeftCorner := floats.Distance(env.state,
						[]float64{(9.0 - math.Sqrt(3)) / 20.0, puddleCenters[1][1] + puddleWidth}, 2)
					distToBottomLeftCorner := floats.Distance(env.state,
						[]float64{puddleCenters[1][0] - puddleWidth, puddleCenters[1][1] - puddleWidth}, 2)
					distToRightSide := puddleCenters[1][0] + puddleWidth - env.state[0]
					dist = floats.Min([]float64{distToTopLeftCorner, distToBottomLeftCorner, distToRightSide})
				}
			}
		}
	}

	reward -= 400.0 * dist
	return reward
}

func (env *Puddleworld) inRange(x, min, max float64) bool {
	return x >= min && x <= max
}

// GetAttributes returns attributes for this environment.
func (env *Puddleworld) GetAttributes() rlglue.Attributes {
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

func (env *Puddleworld) GetInfo(info string, value float64) interface{} {
	return nil
}
