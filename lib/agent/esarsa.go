package agent

import (
	"encoding/json"
	"math"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	"github.com/stellentus/cartpoles/lib/util"
)

// Expected sarsa-lambda with tile coding
type ESarsa struct {
	logger.Debug
	rng   *rand.Rand
	tiler util.MultiTiler

	// weights is a slice of weights for each action
	weights [][]float64
}

func init() {
	Add("esarsa-lambda", NewESarsa)
}

func NewESarsa(logger logger.Debug) (rlglue.Agent, error) {
	return &ESarsa{Debug: logger}, nil
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *ESarsa) Initialize(run uint, expAttr, envAttr rlglue.Attributes) error {
	ss := struct {
		Seed        int64
		EnableDebug bool `json:"enable-debug"`
		NumTiles    int  `json:"tiles"`
		TileSpread  int  `json:"tile-spread"`
	}{
		// These default settings will be used if the config doesn't set these values
		NumTiles:   32,
		TileSpread: 4,
	}

	err := json.Unmarshal(expAttr, &ss)
	if err != nil {
		agent.Message("warning", "agent.ESarsa seed wasn't available: "+err.Error())
		ss.Seed = 0
	}

	agent.rng = rand.New(rand.NewSource(ss.Seed + int64(run))) // Create a new rand source for reproducibility

	scalers := []util.Scaler{
		util.NewScaler(-2.4, 2.4, ss.TileSpread),
		util.NewScaler(-4.0, 4.0, ss.TileSpread),
		util.NewScaler(-(12 * 2 * math.Pi / 360), (12 * 2 * math.Pi / 360), ss.TileSpread),
		util.NewScaler(-3.5, 3.5, ss.TileSpread),
	}

	agent.tiler, err = util.NewMultiTiler(4, ss.NumTiles, scalers)
	if err != nil {
		return err
	}

	agent.weights = make([][]float64, 2) // one weight slice for each action
	agent.weights[0] = make([]float64, agent.tiler.NumberOfIndices())
	agent.weights[1] = make([]float64, agent.tiler.NumberOfIndices())

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *ESarsa) Start(state rlglue.State) rlglue.Action {
	return agent.Step(state, 0)
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *ESarsa) Step(state rlglue.State, reward float64) rlglue.Action {
	_, err := agent.tiler.Tile(state) // Rename the _ to whatever you want it to be. It's a slice of indices of tile activations
	if err != nil {
		agent.Message("err", "agent.ESarsa is acting on garbage state because it couldn't create tiles: "+err.Error())
	}

	panic("Step not implemented")
	return 0
}

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *ESarsa) End(state rlglue.State, reward float64) {
	panic("End not implemented")
}
