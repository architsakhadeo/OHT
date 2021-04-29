package agent

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	"github.com/stellentus/cartpoles/lib/util"
	"gonum.org/v1/gonum/mat"
)

type acSettings struct {
	EnableDebug bool    `json:"enable-debug"`
	Seed        int64   `json:"seed"`
	TotalLogs   uint    `json:"total-logs"`
	NumTilings  int     `json:"tilings"`
	NumTiles    int     `json:"tiles"`
	Gamma       float64 `json:"gamma"`
	ActorAlpha  float64 `json:"actor-alpha"`
	CriticAlpha float64 `json:"critic-alpha"`
	EnvName     string  `json:"env-name"`

	StateDim int `json:"state-len"`

	NumActions int     `json:"number-of-actions"`
	NumDims    int     `json:"number-of-dimensions"`
	WInit      float64 `json:"weight-init"`
}

type ActorCritic struct {
	logger.Debug
	rng *rand.Rand
	acSettings
	tiler                  util.MultiTiler
	oldStateActiveFeatures []float64
	oldAction              rlglue.Action
	oldProbs               mat.Vector
	actor                  Actor
	critic                 Critic
	timesteps              int
}

type Actor struct {
	weight *mat.Dense
	alpha  float64
	rng    *rand.Rand
}

type Critic struct {
	weight *mat.VecDense
	alpha  float64
}

func init() {
	Add("actorcritic", NewActorCritic)
}

func NewActorCritic(logger logger.Debug) (rlglue.Agent, error) {
	return &ActorCritic{Debug: logger}, nil
}

// Init initializes an actor
func (actor *Actor) Init(xdim int, actiondim int, alpha float64, rng *rand.Rand) error {
	actor.weight = mat.NewDense(actiondim, xdim, nil)
	actor.alpha = alpha
	actor.rng = rng
	return nil
}

// Act returns the action and the probabilities of each action given a feature vector x
func (actor *Actor) Act(x []float64) (rlglue.Action, mat.Vector) {
	feat := mat.NewVecDense(len(x), x)
	actiondim, _ := actor.weight.Dims()
	logit := mat.NewVecDense(actiondim, nil)
	// r, c := actor.weight.Dims()
	// fmt.Println("feat dim:", feat.Len(), "weight dim:", r, c)
	logit.MulVec(actor.weight, feat)

	probs := softmax(logit)

	action := sample(probs, actor.rng)

	return action, probs
}

// Update updates the weights of the actor.
func (actor *Actor) Update(x []float64, delta float64, action rlglue.Action, probs mat.Vector) error {

	// calculate the gradient
	adim := probs.Len()
	// r, c := actor.weight.Dims()
	// fmt.Println("probs", adim, "w", r, c)
	dLogPi := mat.NewDense(adim, len(x), nil)

	actionInt, ok := action.(int)
	if !ok {
		return errors.New("action is not an integer")
	}
	dLogPi.SetRow(actionInt, x)

	diag := mat.NewDense(adim, adim, nil)
	for i := 0; i < adim; i++ {
		diag.Set(i, i, probs.AtVec(i))
	}

	tmp := mat.NewDense(adim, len(x), nil)
	tmp.Mul(diag, actor.weight)

	dLogPi.Sub(dLogPi, tmp)

	// do the update
	dLogPi.Scale(actor.alpha*delta, dLogPi)
	actor.weight.Sub(actor.weight, dLogPi)
	return nil
}

// Init initilaizes a critic
func (critic *Critic) Init(xdim int, alpha float64) error {
	critic.weight = mat.NewVecDense(xdim, nil)
	critic.alpha = alpha
	return nil
}

// ValueAt returns the value given a feature vector x
func (critic *Critic) ValueAt(x []float64) float64 {
	feat := mat.NewVecDense(len(x), x)
	v := mat.Dot(critic.weight, feat)
	return v
}

// Update updates the weights of the critic.
func (critic *Critic) Update(x []float64, delta float64) error {
	n := critic.weight.Len()
	update := mat.NewVecDense(n, nil)
	update.AddScaledVec(update, critic.alpha*delta, mat.NewVecDense(n, x))
	critic.weight.AddVec(critic.weight, update)
	return nil
}

// Initialize initializes the ActorCritic Agent following the rlglue framework
func (agent *ActorCritic) Initialize(run uint, expAttr, envAttr rlglue.Attributes) error {
	err := json.Unmarshal(expAttr, &agent.acSettings)
	if err != nil {
		return errors.New("ActorCritic experiment attributes are not valid: " + err.Error())
	}

	err = json.Unmarshal(envAttr, &agent)
	if err != nil {
		agent.Message("ActorCritic agent attributes are not valid: " + err.Error())
	}

	agent.rng = rand.New(rand.NewSource(agent.Seed + int64(run)))

	var stateDim int
	if agent.acSettings.EnvName == "cartpole" {
		agent.NumActions = 2
		stateDim = 4
		scalers := []util.Scaler{
			util.NewScaler(-maxPosition, maxPosition, agent.acSettings.NumTiles),
			util.NewScaler(-maxVelocity, maxVelocity, agent.acSettings.NumTiles),
			util.NewScaler(-maxAngle, maxAngle, agent.acSettings.NumTiles),
			util.NewScaler(-maxAngularVelocity, maxAngularVelocity, agent.acSettings.NumTiles),
		}

		agent.tiler, err = util.NewMultiTiler(stateDim, agent.acSettings.NumTilings, scalers)
		if err != nil {
			return err
		}
	} else if agent.acSettings.EnvName == "acrobot" {
		agent.NumActions = 2 //3
		stateDim = 6
		scalers := []util.Scaler{
			util.NewScaler(-maxFeature1, maxFeature1, agent.acSettings.NumTiles),
			util.NewScaler(-maxFeature2, maxFeature2, agent.acSettings.NumTiles),
			util.NewScaler(-maxFeature3, maxFeature3, agent.acSettings.NumTiles),
			util.NewScaler(-maxFeature4, maxFeature4, agent.acSettings.NumTiles),
			util.NewScaler(-maxFeature5, maxFeature5, agent.acSettings.NumTiles),
			util.NewScaler(-maxFeature6, maxFeature6, agent.acSettings.NumTiles),
		}

		agent.tiler, err = util.NewMultiTiler(stateDim, agent.acSettings.NumTilings, scalers)
		if err != nil {
			return err
		}
	} else if agent.acSettings.EnvName == "puddleworld" {
		agent.NumActions = 4 // 5
		stateDim = 2
		scalers := []util.Scaler{
			util.NewScaler(-maxFeature1, maxFeature1, agent.acSettings.NumTiles),
			util.NewScaler(-maxFeature2, maxFeature2, agent.acSettings.NumTiles),
		}

		agent.tiler, err = util.NewMultiTiler(stateDim, agent.acSettings.NumTilings, scalers)
		if err != nil {
			return err
		}
	} else if agent.acSettings.EnvName == "gridworld" {
		agent.NumActions = 4 // 5
		stateDim = 2
		scalers := []util.Scaler{
			util.NewScaler(minCoord, maxCoord, agent.acSettings.NumTiles),
			util.NewScaler(minCoord, maxCoord, agent.acSettings.NumTiles),
		}

		agent.tiler, err = util.NewMultiTiler(stateDim, agent.acSettings.NumTilings, scalers)
		if err != nil {
			return err
		}
	}

	agent.actor = Actor{}
	err = agent.actor.Init(agent.tiler.NumberOfIndices(), agent.NumActions, agent.ActorAlpha, agent.rng)
	if err != nil {
		return err
	}
	agent.critic = Critic{}
	agent.critic.Init(agent.tiler.NumberOfIndices(), agent.CriticAlpha)
	if err != nil {
		return err
	}

	return nil
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *ActorCritic) Start(state rlglue.State) rlglue.Action {
	xind, err := agent.tiler.Tile(state)
	if err != nil {
		agent.Message("err", "agent.ActorCritic is acting on garbage state because it couldn't create tiles: "+err.Error())
	}

	feat := prepareFeatFromTC(xind, agent.tiler.NumberOfIndices())

	agent.oldAction, agent.oldProbs = agent.actor.Act(feat)
	agent.oldStateActiveFeatures = feat

	agent.timesteps++

	if agent.EnableDebug {
		agent.Message("msg", "start")
	}

	return agent.oldAction
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *ActorCritic) Step(state rlglue.State, reward float64) rlglue.Action {
	xind, err := agent.tiler.Tile(state)
	if err != nil {
		agent.Message("err", "agent.ActorCritic is acting on garbage state because it couldn't create tiles: "+err.Error())
	}

	feat := prepareFeatFromTC(xind, agent.tiler.NumberOfIndices())

	newAction, probs := agent.actor.Act(feat)

	delta := reward + agent.Gamma*agent.critic.ValueAt(feat) - agent.critic.ValueAt(agent.oldStateActiveFeatures)

	agent.actor.Update(feat, delta, agent.oldAction, agent.oldProbs)
	agent.critic.Update(feat, delta)

	agent.oldAction = newAction
	agent.oldProbs = probs
	agent.oldStateActiveFeatures = feat

	agent.timesteps++
	return agent.oldAction
}

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *ActorCritic) End(state rlglue.State, reward float64) {
	xind, err := agent.tiler.Tile(state)
	if err != nil {
		agent.Message("err", "agent.ActorCritic is acting on garbage state because it couldn't create tiles: "+err.Error())
	}

	feat := prepareFeatFromTC(xind, agent.tiler.NumberOfIndices())

	newAction, probs := agent.actor.Act(feat)

	delta := reward + agent.Gamma*agent.critic.ValueAt(feat) - agent.critic.ValueAt(agent.oldStateActiveFeatures)

	agent.actor.Update(feat, delta, agent.oldAction, agent.oldProbs)
	agent.critic.Update(feat, delta)

	agent.oldAction = newAction
	agent.oldProbs = probs
	agent.oldStateActiveFeatures = feat

	agent.timesteps++
}

func (agent *ActorCritic) SaveWeights(basePath string) error {
	return nil
}

func (agent *ActorCritic) GetLearnProg() float64 {
	return float64(0)
}

func (agent *ActorCritic) GetLock() bool {
	return false
}

// softmax return the softmax of the given logit
func softmax(logit mat.Vector) mat.Vector {
	probs := mat.NewVecDense(logit.Len(), nil)
	denom := 0.0
	for i := 0; i < logit.Len(); i++ {
		denom += math.Exp(logit.AtVec(i))
	}

	for i := 0; i < probs.Len(); i++ {
		probs.SetVec(i, math.Exp(logit.AtVec(i)/denom))
	}

	return probs
}

// sample takes the probabilities of each category and return a sample
// from this category using the given rng
func sample(probs mat.Vector, rng *rand.Rand) int {
	dim, _ := probs.Dims()
	cumProbs := make([]float64, dim)
	sum := 0.0
	for i := 0; i < dim; i++ {
		sum += probs.AtVec(i)
		cumProbs[i] = sum
	}

	// random sample from rng
	ret := 0
	n := rng.Float64()
	for i := 0; i < dim; i++ {
		if n <= cumProbs[i] {
			ret = i
			break
		}
	}
	return ret
}

func sliceIntToFloat64(ar []int) []float64 {
	newar := make([]float64, len(ar))
	for i, v := range ar {
		newar[i] = float64(v)
	}
	return newar
}

func prepareFeatFromTC(ind []int, xdim int) []float64 {
	x := make([]float64, xdim)
	for _, i := range ind {
		x[i] = 1.0
	}
	return x
}
