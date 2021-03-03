package state

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

// SensorDriftWrapper is used to wrap an environment, adding sensor drift.
type SensorDriftWrapper struct {
	logger.Debug
	Env rlglue.Environment
	// WrappedEnvAttrs is information loaded from the wrapped environment.
	// The wrapped environment must provide all of the JSON attributes in WrappedEnvAttrs, otherwise Initialize will return an error.
	WrappedEnvAttrs struct {
		// NumAction is the number of actions that can be taken in the environment.
		NumAction int64 `json:"numberOfActions"`
		// StateDim is the dimension of the observation array.
		StateDim int `json:"stateDimension"`
		// StateRange is the range of observations, obtained by (max observation - min observation).
		StateRange []float64 `json:"stateRange"`
	}
	// DriftAttrs is information loaded from the json config file.
	DriftAttrs struct {
		Seed int64 `json:"seed"`
		// DriftScale is the scale of the drift with respect to the state range.
		DriftScale []float64 `json:"driftScale"`
		// SensorLife is the number of time-steps after which the probability of drift will become nearly 1
		SensorLife []float64 `json:"sensorLife"`
		// DriftProb is the probability scale, if it's less than 0, drift occurs with prob=1, if it's between 0 and 1, the max prob is scaled by this value.
		DriftProb []float64 `json:"driftProb"`
	}
	// noise is the zero-mean Gaussian noise applied to the observation at each time-step.
	noise []float64
	// noiseStd is the standard deviation of the noise generated at each time-step.
	noiseStd []float64
	// noiseMax is the maximum noise value used for clamping the noise and observation.
	noiseMax []float64
	// sensorSteps is the number of sensor drift time-steps.
	sensorSteps int64
	noiseFns    []func(int) float64
	rng         *rand.Rand
	// stateProcessFn is the function that processes the state from the environment and passes the output to the agent.
	stateProcessFns []func(float64, int) float64
}

func init() {
	Add("sensor-drift", NewSensorDriftWrapper)
}

func NewSensorDriftWrapper(logger logger.Debug, env rlglue.Environment) (rlglue.Environment, error) {
	return &SensorDriftWrapper{Debug: logger, Env: env}, nil
}

// Initialize configures the environment with the provided parameters and resets any internal state.
func (wrapper *SensorDriftWrapper) Initialize(run uint, attr rlglue.Attributes) error {
	err := json.Unmarshal(attr, &wrapper.DriftAttrs)
	if err != nil {
		err = errors.New("environment.SensorDriftWrapper settings error: " + err.Error())
		wrapper.Message("err", err)
		return err
	}
	if wrapper.DriftAttrs.SensorLife == nil || wrapper.DriftAttrs.DriftProb == nil || wrapper.DriftAttrs.DriftScale == nil {
		err = errors.New("environment.SensorDriftWrapper settings error: Invalid sensor drift attribute(s) in the config file")
		wrapper.Message("err", err)
		return err
	}
	wrapper.rng = rand.New(rand.NewSource(wrapper.DriftAttrs.Seed + int64(run)))

	envAttr := wrapper.Env.GetAttributes()
	err = json.Unmarshal(envAttr, &wrapper.WrappedEnvAttrs)
	if err != nil {
		err = errors.New("environment.SensorDriftWrapper settings error: " + err.Error())
		wrapper.Message("err", err)
		return err
	}
	if wrapper.WrappedEnvAttrs.NumAction == 0 || wrapper.WrappedEnvAttrs.StateDim == 0 || wrapper.WrappedEnvAttrs.StateRange == nil {
		err = errors.New("environment.SensorDriftWrapper settings error: Invalid attribute(s) from the wrapped environment")
		wrapper.Message("err", err)
		return err
	}

	// Preprocess drift attributes.
	wrapper.DriftAttrs.SensorLife, err = wrapper.attrPreprocess(wrapper.DriftAttrs.SensorLife)
	if err != nil {
		return err
	}
	wrapper.DriftAttrs.DriftProb, err = wrapper.attrPreprocess(wrapper.DriftAttrs.DriftProb)
	if err != nil {
		return err
	}
	wrapper.DriftAttrs.DriftScale, err = wrapper.attrPreprocess(wrapper.DriftAttrs.DriftScale)
	if err != nil {
		return err
	}

	wrapper.noise = make([]float64, wrapper.WrappedEnvAttrs.StateDim)
	wrapper.noiseStd = make([]float64, wrapper.WrappedEnvAttrs.StateDim)
	for i := range wrapper.noiseStd {
		wrapper.noiseStd[i] = wrapper.WrappedEnvAttrs.StateRange[i] * wrapper.DriftAttrs.DriftScale[i]
	}
	wrapper.noiseMax = make([]float64, wrapper.WrappedEnvAttrs.StateDim)
	for i := range wrapper.noiseMax {
		wrapper.noiseMax[i] = wrapper.WrappedEnvAttrs.StateRange[i] / 2
	}
	wrapper.noiseFns = make([]func(int) float64, wrapper.WrappedEnvAttrs.StateDim)
	for i := range wrapper.DriftAttrs.DriftProb {
		if wrapper.DriftAttrs.DriftProb[i] < 0 {
			wrapper.noiseFns[i] = wrapper.gaussNoise
		} else {
			wrapper.noiseFns[i] = wrapper.probGaussNoise
		}
	}
	wrapper.stateProcessFns = make([]func(float64, int) float64, wrapper.WrappedEnvAttrs.StateDim)
	for i := range wrapper.DriftAttrs.DriftScale {
		if wrapper.DriftAttrs.DriftScale[i] != 0 {
			wrapper.stateProcessFns[i] = wrapper.stateProcess
		} else {
			wrapper.stateProcessFns[i] = wrapper.stateRaw
		}
	}

	wrapper.Message("msg", "environment.SensorDriftWrapper Initialize",
		"drift scale", wrapper.DriftAttrs.DriftScale,
		"sensor life", wrapper.DriftAttrs.SensorLife,
		"drift probability", wrapper.DriftAttrs.DriftProb)
	return nil
}

// Start returns an initial observation.
func (wrapper *SensorDriftWrapper) Start(randomizeStartStateCondition bool) (rlglue.State, string) {
	return wrapper.Env.Start(randomizeStartStateCondition)
}

// Step takes an action and provides the resulting reward, the new observation, and whether the state is terminal.
// For this continuous environment, it's only terminal if the action was invalid.
func (wrapper *SensorDriftWrapper) Step(act rlglue.Action, randomizeStartStateCondition bool) (rlglue.State, float64, bool, string) {
	state, reward, done, _ := wrapper.Env.Step(act, randomizeStartStateCondition)
	wrapper.sensorSteps++
	for i := 0; i < wrapper.WrappedEnvAttrs.StateDim; i++ {
		state[i] = wrapper.stateProcessFns[i](state[i], i)
	}
	return state, reward, done, ""
}

// GetAttributes returns attributes for this environment.
func (wrapper *SensorDriftWrapper) GetAttributes() rlglue.Attributes {
	return wrapper.Env.GetAttributes()
}

func (wrapper *SensorDriftWrapper) attrPreprocess(attr []float64) ([]float64, error) {
	if len(attr) == 1 {
		at := attr[0]
		attr = []float64{at, at, at, at}
	} else if len(attr) != 4 {
		err := errors.New("environment.SensorDriftWrapper settings error: Invalid sensor drift attribute(s) in the config file")
		wrapper.Message("err", err)
		return attr, err
	}
	return attr, nil
}

func (wrapper *SensorDriftWrapper) stateProcess(state float64, idx int) float64 {
	wrapper.noise[idx] = wrapper.clamp(wrapper.noise[idx]+wrapper.noiseFns[idx](idx), idx)
	state = wrapper.clamp(state+wrapper.noise[idx], idx)
	return state
}

func (wrapper *SensorDriftWrapper) stateRaw(state float64, idx int) float64 {
	return state
}

func (wrapper *SensorDriftWrapper) gaussNoise(idx int) float64 {
	return wrapper.rng.NormFloat64() * wrapper.noiseStd[idx]
}

func (wrapper *SensorDriftWrapper) probGaussNoise(idx int) float64 {
	var noise float64
	probDrift := wrapper.logisticProb(idx)
	if sample := wrapper.rng.Float64(); sample < probDrift {
		noise = wrapper.rng.NormFloat64() * wrapper.noiseStd[idx]
	}
	return noise
}

func (wrapper *SensorDriftWrapper) logisticProb(idx int) float64 {
	return wrapper.DriftAttrs.DriftProb[idx] / (1 + math.Exp(-(float64(wrapper.sensorSteps)-wrapper.DriftAttrs.SensorLife[idx]/2)/(wrapper.DriftAttrs.SensorLife[idx]/10)))
}

func (wrapper *SensorDriftWrapper) clamp(x float64, idx int) float64 {
	return math.Max(-wrapper.noiseMax[idx], math.Min(x, wrapper.noiseMax[idx]))
}
