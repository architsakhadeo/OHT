package environment

import (
	"encoding/json"
	"math"
	"math/rand"

	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

const (
	gravity               = 9.8
	masscart              = 1.0
	masspole              = 0.1
	totalMass             = (masspole + masscart)
	length                = 0.5 // half-length
	polemassLength        = (masspole * length)
	forceMag              = 10.0
	tau                   = 0.02
	thetaRhresholdRadians = (12 * 2 * math.Pi / 360)
	xThreshold            = 2.4
	stateMin              = -0.05
	stateMax              = 0.05
)

type Cartpole struct {
	logger.Debug
	Seed            int64     `json:"seed"`
	Delays          int       `json:"delays"`
	PercentNoise    []float64 `json:"percent_noise"`
	state           rlglue.State
	stepsBeyondDone int
	rng             *rand.Rand
}

func init() {
	Add("cartpole", NewCartpole)
}

func NewCartpole(logger logger.Debug) (rlglue.Environment, error) {
	return &Cartpole{Debug: logger}, nil
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (env *Cartpole) Initialize(attr rlglue.Attributes) error {
	env.PercentNoise = make([]float64, 4)
	err := json.Unmarshal(attr, &env)
	if err != nil {
		env.Message("warning", "environment.Cartpole seed wasn't available")
		env.Seed = 0
	}
	env.rng = rand.New(rand.NewSource(env.Seed)) // Create a new rand source for reproducibility

	env.state = make(rlglue.State, 4)
	env.stepsBeyondDone = -1

	env.Message("msg", "environment.Cartpole Initialize", "seed", env.Seed, "delays", env.Delays, "percent noise", env.PercentNoise)

	return nil
}

func (env *Cartpole) noisyState() rlglue.State {
	stateLowerBound := []float64{-2.4, -4.0, -(12 * 2 * math.Pi / 360), -3.5}
	stateUpperBound := []float64{2.4, 4.0, (12 * 2 * math.Pi / 360), 3.5}

	state := make(rlglue.State, 4)
	for i := range state {
		state[i] = env.state[i] + env.randFloat(env.PercentNoise[i]*stateLowerBound[i], env.PercentNoise[i]*stateUpperBound[i])
	}
	return state
}

func (env *Cartpole) randomizeState() {
	for i := range env.state {
		env.state[i] = env.randFloat(stateMin, stateMax)
	}
}

func (env *Cartpole) randFloat(min, max float64) float64 {
	return env.rng.Float64()*(max-min) + min
}

// Start returns an initial observation.
func (env *Cartpole) Start() rlglue.State {
	env.randomizeState()
	env.stepsBeyondDone = -1
	return env.state
}

// Step takes an action and provides the resulting reward, the new observation, and whether the state is terminal.
// For this continuous environment, it's only terminal if the action was invalid.
func (env *Cartpole) Step(act rlglue.Action) (rlglue.State, float64, bool) {
	x, xDot, theta, thetaDot := env.state[0], env.state[1], env.state[2], env.state[3]

	force := forceMag
	// If the action is anything other than exactly 0, consider the action to have been the "move right" action.
	if act == 0 {
		force *= -1
	}

	costheta := math.Cos(theta)
	sintheta := math.Sin(theta)

	temp := (force + polemassLength*thetaDot*thetaDot*sintheta) / totalMass
	thetaacc := (gravity*sintheta - costheta*temp) / (length * (4.0/3.0 - masspole*costheta*costheta/totalMass))
	xacc := temp - polemassLength*thetaacc*costheta/totalMass

	// euler
	x = x + tau*xDot
	xDot = xDot + tau*xacc
	theta = theta + tau*thetaDot
	thetaDot = thetaDot + tau*thetaacc

	env.state = rlglue.State{x, xDot, theta, thetaDot}

	done := (x < -xThreshold) || (x > xThreshold) || (theta < -thetaRhresholdRadians) || (theta > thetaRhresholdRadians)

	var reward float64
	if !done {
	} else if env.stepsBeyondDone == -1 {
		env.stepsBeyondDone = 0
		reward = -1.0
		env.randomizeState()
	} else {
		env.stepsBeyondDone += 1
		reward = -1.0
		env.randomizeState()
	}

	// Add noise to state to get observations
	observations := env.noisyState()

	return observations, reward, done
}

// GetAttributes returns attributes for this environment.
func (env *Cartpole) GetAttributes() rlglue.Attributes {
	return rlglue.Attributes(`{"numberOfActions":2}`)
}
