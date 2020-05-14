package environment

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

// const (

// )

// Example is just a dumb test environment.
// state is adjusted by the action. Valid values are integers [0, ExampleNumberOfActions).
// Reward is equal to the new state.
// When |state| is >= ExampleStateMax, it's reset to 0.
type SensorDriftWrapper struct {
	logger.Debug
	Env         rlglue.Environment
	Seed        int64     `json:"seed"`
	NumAction   int64     `json:"numberOfActions"`
	StateDim    int       `json:"stateDimension"`
	StateRange  []float64 `json:"stateRange"`
	Noise       []float64
	NoiseStd    []float64
	NoiseMax    []float64
	sensorSteps int64
	Sweep       struct {
		DriftScale float64   `json:"driftScale"`
		SensorLife []float64 `json:"sensorLife"`
		DriftProb  float64   `json:"driftProb"`
	} `json:"sweep"`
	NoiseFn func() []float64
	rng     *rand.Rand
}

func NewSensorDriftWrapper(logger logger.Debug, env rlglue.Environment) (rlglue.Environment, error) {
	return &SensorDriftWrapper{Debug: logger, Env: env}, nil
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (wrapper *SensorDriftWrapper) Initialize(run uint, attr rlglue.Attributes) error {
	wrapper.rng = rand.New(rand.NewSource(wrapper.Seed + int64(run)))
	// wrapper.Env = env
	// envAttr := env.GetAttributes()
	err := json.Unmarshal(attr, &wrapper)
	if err != nil {
		err = errors.New("environment.SensorDriftWrapper settings error: " + err.Error())
		wrapper.Message("err", err)
		return err
	}
	envAttr := wrapper.Env.GetAttributes()
	err = json.Unmarshal(envAttr, &wrapper)
	if err != nil {
		err = errors.New("environment.SensorDriftWrapper settings error: " + err.Error())
		wrapper.Message("err", err)
		return err
	}

	wrapper.Noise = make([]float64, wrapper.StateDim)
	wrapper.NoiseStd = make([]float64, wrapper.StateDim)
	for i := range wrapper.NoiseStd {
		wrapper.NoiseStd[i] = wrapper.StateRange[i] / wrapper.Sweep.DriftScale
	}
	wrapper.NoiseMax = make([]float64, wrapper.StateDim)
	for i := range wrapper.NoiseMax {
		wrapper.NoiseMax[i] = wrapper.StateRange[i] / 2
	}
	if wrapper.Sweep.DriftProb < 0 {
		wrapper.NoiseFn = wrapper.gaussNoise
	} else {
		wrapper.NoiseFn = wrapper.probGaussNoise
	}

	wrapper.Message("msg", "environment.SensorDriftWrapper Initialize",
		"drift scale", wrapper.Sweep.DriftScale,
		"sensor life", wrapper.Sweep.SensorLife,
		"drift prob", wrapper.Sweep.DriftProb)
	return nil
}

// Start returns an initial observation.
func (wrapper *SensorDriftWrapper) Start() rlglue.State {
	return wrapper.Env.Start()
}

// Step takes an action and provides the resulting reward, the new observation, and whether the state is terminal.
// For this continuous environment, it's only terminal if the action was invalid.
func (wrapper *SensorDriftWrapper) Step(act rlglue.Action) (rlglue.State, float64, bool) {
	state, reward, done := wrapper.Env.Step(act)
	state = wrapper.stateProcess(state)
	return state, reward, done
}

// GetAttributes returns attributes for this environment.
func (wrapper *SensorDriftWrapper) GetAttributes() rlglue.Attributes {
	return wrapper.Env.GetAttributes()
}

func (wrapper *SensorDriftWrapper) stateProcess(state rlglue.State) rlglue.State {
	noiseNew := wrapper.NoiseFn()
	for i := 0; i < wrapper.StateDim; i++ {
		wrapper.Noise[i] = wrapper.clamp(wrapper.Noise[i]+noiseNew[i], -wrapper.NoiseMax[i], wrapper.NoiseMax[i])
		state[i] = wrapper.clamp(state[i]+wrapper.Noise[i], -wrapper.NoiseMax[i], wrapper.NoiseMax[i])
	}
	return state
}

func (wrapper *SensorDriftWrapper) gaussNoise() []float64 {
	noise := make([]float64, wrapper.StateDim)
	for i := range noise {
		noise[i] = wrapper.rng.NormFloat64() * wrapper.NoiseStd[i]
	}
	return noise
}

func (wrapper *SensorDriftWrapper) probGaussNoise() []float64 {
	wrapper.sensorSteps++
	noise := make([]float64, wrapper.StateDim)
	probDrift := wrapper.logisticProb()
	for i := range probDrift {
		if sample := wrapper.rng.Float64(); sample < probDrift[i] {
			noise[i] = wrapper.rng.NormFloat64() * wrapper.NoiseStd[i]
		}
	}
	return noise
}

func (wrapper *SensorDriftWrapper) logisticProb() []float64 {
	cdf := make([]float64, wrapper.StateDim)
	for i := range cdf {
		cdf[i] = wrapper.Sweep.DriftProb / (1 + math.Exp(-(float64(wrapper.sensorSteps)-wrapper.Sweep.SensorLife[i]/2)/(wrapper.Sweep.SensorLife[i]/10)))
	}
	return cdf
}

func (wrapper *SensorDriftWrapper) clamp(x, min, max float64) float64 {
	return math.Max(min, math.Min(x, max))
}
