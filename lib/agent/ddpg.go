package agent

import (
	"encoding/json"
	"errors"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/util/loss"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	ao "github.com/stellentus/cartpoles/lib/util/array-opr"
	"github.com/stellentus/cartpoles/lib/util/buffer"
	"github.com/stellentus/cartpoles/lib/util/network"
	"github.com/stellentus/cartpoles/lib/util/normalizer"
	"github.com/stellentus/cartpoles/lib/util/optimizer"
)

type ddpgSettings struct {
	Seed        int64
	EnableDebug bool `json:"enable-debug"`

	ActionDim           int     `json:"action-dimension"`
	StateContainsReplay bool    `json:"state-contains-replay"`
	Gamma               float64 `json:"gamma"`

	ActorHidden  []int   `json:"actor-hidden"`
	ActionStd    float64 `json:"action-std"`
	CriticHidden []int   `json:"critic-hidden"`
	Alpha        float64 `json:"alpha"`
	Sync         int     `json:"ddpg-sync"`
	Decay        float64 `json:"ddpg-decay"`
	Momentum     float64 `json:"ddpg-momentum"`
	AdamBeta1    float64 `json:"ddpg-adamBeta1"`
	AdamBeta2    float64 `json:"ddpg-adamBeta2"`
	AdamEps      float64 `json:"ddpg-adamEps"`

	Bsize int    `json:"buffer-size"`
	Btype string `json:"buffer-type"`

	StateDim   int  `json:"state-len"`
	BatchSize  int  `json:"dqn-batch"`
	IncreaseBS bool `json:"increasing-batch"`

	StateRange []float64 `json:"StateRange"`

	OptName string `json:"optimizer"`
}

type Ddpg struct {
	logger.Debug
	rng        *rand.Rand
	lastAction interface{}
	lastState  rlglue.State

	ddpgSettings

	updateNum int
	learning  bool

	nml normalizer.Normalizer
	bf  *buffer.Buffer

	CriticLearning network.Network
	CriticTarget   network.Network
	ActorLearning  network.Network
	ActorTarget    network.Network

	CriticOpt optimizer.Optimizer
	ActorOpt  optimizer.Optimizer
}

func init() {
	Add("ddpg", NewDdpg)
}

func NewDdpg(logger logger.Debug) (rlglue.Agent, error) {
	return &Ddpg{Debug: logger}, nil
}

func (agent *Ddpg) Initialize(run uint, expAttr, envAttr rlglue.Attributes) error {
	err := json.Unmarshal(expAttr, &agent.ddpgSettings)
	if err != nil {
		return errors.New("DDPG agent attributes were not valid: " + err.Error())
	}

	err = json.Unmarshal(envAttr, &agent)
	if err != nil {
		agent.Message("err", "envAttr unmarshal not valid: "+err.Error())
	}
	agent.rng = rand.New(rand.NewSource(agent.Seed + int64(run))) // Create a new rand source for reproducibility

	if agent.EnableDebug {
		agent.Message("msg", "agent.Example Initialize", "seed", agent.Seed, "ActionDimension", agent.ActionDim)
	}
	agent.bf = buffer.NewBuffer()
	agent.bf.Initialize(agent.Btype, agent.Bsize, agent.StateDim, agent.Seed+int64(run))

	agent.nml = normalizer.Normalizer{agent.StateDim, agent.StateRange}

	// NN: Graph Construction
	// NN: Weight Initialization
	agent.CriticLearning = network.CreateNetwork(agent.StateDim+agent.ActionDim, agent.CriticHidden, 1, agent.Alpha,
		agent.Decay, agent.Momentum, agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps)
	agent.CriticTarget = network.CreateNetwork(agent.StateDim+agent.ActionDim, agent.CriticHidden, 1, agent.Alpha,
		agent.Decay, agent.Momentum, agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps)
	agent.ActorLearning = network.CreateNetwork(agent.StateDim, agent.ActorHidden, agent.ActionDim, agent.Alpha,
		agent.Decay, agent.Momentum, agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps)
	agent.ActorTarget = network.CreateNetwork(agent.StateDim, agent.ActorHidden, agent.ActionDim, agent.Alpha,
		agent.Decay, agent.Momentum, agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps)
	agent.updateNum = 0

	if agent.OptName == "Adam" {
		agent.ActorOpt = new(optimizer.Adam)
		agent.ActorOpt.Init(agent.Alpha, []float64{agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps}, agent.StateDim, agent.ActorHidden, agent.ActionDim)
		agent.CriticOpt = new(optimizer.Adam)
		agent.CriticOpt.Init(agent.Alpha, []float64{agent.AdamBeta1, agent.AdamBeta2, agent.AdamEps}, agent.StateDim+agent.ActionDim, agent.CriticHidden, 1)
	} else if agent.OptName == "Sgd" {
		agent.ActorOpt = new(optimizer.Sgd)
		agent.ActorOpt.Init(agent.Alpha, []float64{agent.Momentum}, agent.StateDim, agent.ActorHidden, agent.ActionDim)
		agent.CriticOpt = new(optimizer.Sgd)
		agent.CriticOpt.Init(agent.Alpha, []float64{agent.Momentum}, agent.StateDim+agent.ActionDim, agent.CriticHidden, 1)
	} else {
		errors.New("Optimizer NotImplemented")
	}

	return nil
}

func (agent *Ddpg) Start(oristate rlglue.State) rlglue.Action {
	state := make([]float64, agent.StateDim)
	copy(state, oristate)

	state = agent.nml.MeanZeroNormalization(state)
	agent.lastState = state
	act := agent.Policy(state)
	agent.lastAction = act

	if agent.EnableDebug {
		agent.Message("msg", "start")
	}
	return rlglue.Action(act)
}

func (agent *Ddpg) Step(oristate rlglue.State, reward float64) rlglue.Action {
	state := make([]float64, agent.StateDim)
	copy(state, oristate)
	state = agent.nml.MeanZeroNormalization(state)
	agent.bf.Feed(agent.lastState, agent.lastAction, state, reward, agent.Gamma)
	//agent.stepNum = agent.stepNum + 1
	agent.Update()
	agent.lastState = state
	agent.lastAction = agent.Policy(state)
	act := rlglue.Action(agent.lastAction)

	if agent.EnableDebug {
		if agent.StateContainsReplay {
			agent.Message("msg", "step", "state", state[0], "reward", reward, "action", act, "expected action", state[1])
		} else {
			agent.Message("msg", "step", "state", state, "reward", reward, "action", act)
		}
	}
	return act
}

func (agent *Ddpg) End(state rlglue.State, reward float64) {
	agent.bf.Feed(agent.lastState, agent.lastAction, state, reward, float64(0)) // gamma=0
	agent.Update()
	if agent.EnableDebug {
		agent.Message("msg", "end", "state", state, "reward", reward)
	}
}

func (agent *Ddpg) Update() {
	if agent.updateNum%agent.Sync == 0 {
		// NN: Synchronization
		agent.CriticTarget = network.Synchronization(agent.CriticLearning, agent.CriticTarget)
		agent.ActorTarget = network.Synchronization(agent.ActorLearning, agent.ActorTarget)
	}

	lastStates, lastActions, states, rewards, gammas := agent.bf.Sample(agent.BatchSize)

	// Critic: Weight update
	nextA := agent.ActorTarget.Predict(states)
	temp := ao.Concatenate(states, nextA)
	nextQ := agent.CriticTarget.Predict(temp)
	//y := ao.BitwiseAdd2D(
	//	ao.ReSize1DA64(rewards, len(rewards), 1),
	//	ao.BitwiseMulti2D(ao.ReSize1DA64(gammas, len(gammas), 1), nextQ))
	//temp = ao.Concatenate(lastStates, ao.ReSize1DA64(lastActions, len(lastActions), 1))
	y := ao.BitwiseAdd2D(rewards, ao.BitwiseMulti2D(gammas, nextQ))
	temp = ao.Concatenate(lastStates, lastActions)
	currentBufferQ := agent.CriticLearning.Forward(temp)
	criticLoss := loss.MseLossDeriv(y, currentBufferQ)
	agent.CriticLearning.Backward(criticLoss, agent.CriticOpt)

	// Actor: Weight update
	currentActor := agent.ActorLearning.Forward(lastStates)
	temp = ao.Concatenate(lastStates, currentActor)
	currentActorQ := agent.CriticLearning.Predict(temp)
	negQ := ao.A64ArrayMulti2D(-1.0, currentActorQ)
	agent.ActorLearning.Backward(negQ, agent.ActorOpt)

	agent.updateNum += 1
}

// Choose action
func (agent *Ddpg) Policy(state rlglue.State) float64 {
	// NN: choose action
	inputS := make([][]float64, 1)
	inputS[0] = state
	mu := agent.ActorLearning.Predict(inputS)[0]
	action := mu[0] + rand.NormFloat64()*agent.ActionStd
	return action
}

func (agent *Ddpg) GetLock() bool {
	return false
}

func (agent *Ddpg) SaveWeights(basePath string) error {
	return nil
}

func (agent *Ddpg) GetLearnProg() string {
	return "0"
}
