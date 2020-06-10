package agent

import (
	"encoding/json"
	"math/rand"
	"io/ioutil"
	"log"
	"os/exec"
	"fmt"
	// "strconv"
	// "reflect"
	
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	// "github.com/tensorflow/tensorflow/tensorflow/go/op"

	ao "github.com/stellentus/cartpoles/lib/utils/array-opr"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	"github.com/stellentus/cartpoles/lib/utils/buffer"
)


type Model struct {
	graph *tf.Graph
	sess  *tf.Session

	behTruth tf.Output
	behOut tf.Output
	tarOut tf.Output

	behIn  tf.Output
	tarIn  tf.Output
	gammaIn tf.Output
	rewardIn tf.Output

	initOp         *tf.Operation
	trainOp        *tf.Operation

	syncOp1	*tf.Operation
	syncOp2	*tf.Operation
	syncOp3	*tf.Operation
}

type Dqn struct {
	logger.Debug
	rng					*rand.Rand
	lastAction          int
	lastState			rlglue.State
	EnableDebug         bool
	NumberOfActions     int `json:"numberOfActions"`
	StateContainsReplay bool `json:"state-contains-replay"`
	Gamma				float64 `json:"gamma"`
	Epsilon				float64 `json:"epsilon"`
	Hidden				int `json:"dqn-hidden"`
	Layer				int `json:"dqn-ly"`
	Alpha				int `json:"alpha"`
	Sync				int `json:"dqn-sync"`
	updateNum			int
	
	bf					*buffer.Buffer
	Bsize				int `json:"buffer-size"`
	Btype				string `json:"buffer-type"`

	StateDim			int `json:"state-len"`
	BatchSize			int `json:"dqn-batch"`

	valueNet			*Model	
}

func init() {
	Add("dqn", NewDqn)
}

func NewDqn(logger logger.Debug) (rlglue.Agent, error) {
	return &Dqn{Debug: logger}, nil
}

func (agent *Dqn) Initialize(expAttr, envAttr rlglue.Attributes) error {
	var ss struct {
		Seed        int64
		EnableDebug bool `json:"enable-debug"`

		NumberOfActions     int `json:"numberOfActions"`
		StateContainsReplay bool `json:"state-contains-replay"`
		Gamma				float64 `json:"gamma"`
		Epsilon				float64 `json:"epsilon"`
		Hidden				int `json:"dqn-hidden"`
		Layer				int `json:"dqn-ly"`
		Alpha				int `json:"alpha"`
		Sync				int `json:"dqn-sync"`

		Bsize				int `json:"buffer-size"`
		Btype				string `json:"buffer-type"`

		StateDim			int `json:"state-len"`
		BatchSize			int `json:"dqn-batch"`
	}
	err := json.Unmarshal(expAttr, &ss)
	if err != nil {
		agent.Message("warning", "agent.Example seed wasn't available: "+err.Error())
		ss.Seed = 0
	}
	agent.EnableDebug = ss.EnableDebug
	agent.NumberOfActions = ss.NumberOfActions
	agent.StateContainsReplay = ss.StateContainsReplay
	agent.Gamma = ss.Gamma
	agent.Epsilon = ss.Epsilon
	agent.Hidden = ss.Hidden
	agent.Layer = ss.Layer
	agent.Alpha = ss.Alpha
	agent.Sync = ss.Sync
	agent.Bsize = ss.Bsize
	agent.Btype = ss.Btype
	agent.StateDim = ss.StateDim
	agent.BatchSize = ss.BatchSize

	err = json.Unmarshal(envAttr, &agent)
	if err != nil {
		agent.Message("err", "agent.Example number of Actions wasn't available: "+err.Error())
	}
	agent.rng = rand.New(rand.NewSource(ss.Seed)) // Create a new rand source for reproducibility
	// agent.lastAction = rng.Intn(agent.NumberOfActions)

	if agent.EnableDebug {
		agent.Message("msg", "agent.Example Initialize", "seed", ss.Seed, "numberOfActions", agent.NumberOfActions)
	}
	agent.bf = buffer.NewBuffer()
	agent.bf.Initialize(agent.Btype, agent.Bsize, agent.StateDim)

	graphDef := "data/nn/graph.pb"
	cmd := exec.Command("python", "-c", "import lib.utils.network.vanilla; lib.utils.network.vanilla.graph_construction('"+graphDef+"')")
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(fmt.Sprint(err) + ": " + string(output))
		return err
	}
	
	log.Print("Loading graph")
	agent.valueNet = NewModel(graphDef)
	
	if _, err := agent.valueNet.sess.Run(nil, nil, []*tf.Operation{agent.valueNet.initOp}); err != nil {
		panic(err)
	}
	agent.updateNum = 0

	return nil
}

func NewModel(graphDefFilename string) *Model {
	graphDef, err := ioutil.ReadFile(graphDefFilename)
	if err != nil {
		log.Fatal("Failed to read ", graphDefFilename,": ", err)
	}
	graph := tf.NewGraph()
	if err = graph.Import(graphDef, ""); err != nil {
		log.Fatal("Invalid GraphDef?", err)
	}
	sess, err := tf.NewSession(graph, nil)

	if err != nil {
		panic(err)
	}
	return &Model {
		graph: graph,
		sess: sess,
		initOp:  graph.Operation("init"),
		trainOp: graph.Operation("beh_train"),
		behIn: graph.Operation("beh_in").Output(0),
		behTruth: graph.Operation("beh_truth").Output(0),
		behOut: graph.Operation("beh_out").Output(0),
		tarIn: graph.Operation("target_in").Output(0),
		gammaIn: graph.Operation("gamma").Output(0),
		rewardIn: graph.Operation("reward").Output(0),
		// tarOut: graph.Operation("target_out").Output(0),
		tarOut: graph.Operation("target").Output(0),
		syncOp1: graph.Operation("set1"),
		syncOp2: graph.Operation("set2"),
		syncOp3: graph.Operation("set3"),
	}
}

// Start provides an initial observation to the agent and returns the agent's action.
func (agent *Dqn) Start(state rlglue.State) rlglue.Action {
	if agent.EnableDebug {
		agent.Message("msg", "start")
	}
	act := agent.Step(state, 0)
	return act
}

// Step provides a new observation and a reward to the agent and returns the agent's next action.
func (agent *Dqn) Step(state rlglue.State, reward float64) rlglue.Action {
	agent.Feed(agent.lastState, agent.lastAction, state, reward, agent.Gamma)
	agent.Update()
	// agent.lastAction = agent.Policy(state)
	agent.lastState = state
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

// End informs the agent that a terminal state has been reached, providing the final reward.
func (agent *Dqn) End(state rlglue.State, reward float64) {
	if agent.EnableDebug {
		agent.Message("msg", "end", "state", state, "reward", reward)
	}
}

func (agent *Dqn) Feed(lastS rlglue.State, lastA int, state rlglue.State, reward float64, gamma float64) {
	agent.bf.Feed(lastS, lastA, state, reward, gamma)
}

func (agent *Dqn) Update() {
	
	if agent.updateNum % agent.Sync == 0 {
		agent.valueNet.sess.Run(nil, nil, []*tf.Operation{agent.valueNet.syncOp1})
		agent.valueNet.sess.Run(nil, nil, []*tf.Operation{agent.valueNet.syncOp2})
		agent.valueNet.sess.Run(nil, nil, []*tf.Operation{agent.valueNet.syncOp3})
	}

	samples64 := agent.bf.Sample(agent.BatchSize)
	// lastStates := samples[:][:agent.StateDim]
	// lastActions := samples[:][agent.StateDim]
	// states := samples[:][agent.StateDim+1: agent.StateDim*2+1]
	// rewards := samples[:][agent.StateDim*2+1]
	// gammas := samples[:][agent.StateDim*2+2]
	
	samples := ao.A64To32_2d(samples64)
	lastStates := ao.Index2d(samples, 0, len(samples), 0, agent.StateDim)
	lastActions := ao.Index2d(samples, 0, len(samples), agent.StateDim, agent.StateDim+1)
	states := ao.Index2d(samples, 0, len(samples), agent.StateDim+1, agent.StateDim*2+1)
	rewards := ao.Index2d(samples, 0, len(samples), agent.StateDim*2+1, agent.StateDim*2+2)
	gammas := ao.Index2d(samples, 0, len(samples), agent.StateDim*2+2, agent.StateDim*2+3)
	
	statesT, err := tf.NewTensor(states)
	if err != nil {
		panic(err)
	}
	// feeds := map[tf.Output]*tf.Tensor{agent.valueNet.tarIn: statesT}
	// fetch := []tf.Output{agent.valueNet.tarOut}
	// qNext, err := agent.valueNet.sess.Run(feeds, fetch, nil)
	// if err != nil {
	// 	panic(err)
	// }
	// fmt.Print(qNext, "\n")
	// qNext := ao.RowIndexMax(qNextAll[0].Value().([][]float32))
	// target := ao.BitwiseAdd(ao.A32Col(rewards, 0), ao.BitwiseMulti(qNext, ao.A32Col(gammas, 0)))
	rewardT, _ := tf.NewTensor(rewards)
	gammaT, _ := tf.NewTensor(gammas)
	feeds := map[tf.Output]*tf.Tensor{agent.valueNet.tarIn: statesT, agent.valueNet.gammaIn: gammaT, agent.valueNet.rewardIn: rewardT}
	fetch := []tf.Output{agent.valueNet.tarOut}
	target, err := agent.valueNet.sess.Run(feeds, fetch, nil)

	lastStatesT, _ := tf.NewTensor(lastStates)
	feeds = map[tf.Output]*tf.Tensor{agent.valueNet.behIn: lastStatesT}
	fetch = []tf.Output{agent.valueNet.behOut}
	qAll, _ := agent.valueNet.sess.Run(feeds, fetch, nil)
	q := ao.RowIndexFloat(qAll[0].Value().([][]float32), ao.A32ToInt(ao.A32Col(lastActions, 0)))
	
	predictionT, _ := tf.NewTensor(q)
	targetT, _ := tf.NewTensor(target)
	feeds = map[tf.Output]*tf.Tensor{
		agent.valueNet.behOut:  predictionT,
		agent.valueNet.behTruth: targetT,
	}
	agent.valueNet.sess.Run(feeds, nil, []*tf.Operation{agent.valueNet.trainOp})

	agent.updateNum += 1
}

// Choose action
func (agent *Dqn) Policy(state rlglue.State) int {
	var idx int
	if rand.Float64() < agent.Epsilon {
		idx = agent.rng.Intn(agent.NumberOfActions)
	} else {
		lastStatesT, _ := tf.NewTensor(agent.lastState)
		feeds := map[tf.Output]*tf.Tensor{agent.valueNet.behOut: lastStatesT}
		fetch := []tf.Output{agent.valueNet.behOut}
		qAll, _ := agent.valueNet.sess.Run(feeds, fetch, nil)
		_, idx = ao.ArrayMax(qAll[0].Value().([]float32))	
	}
	return idx
}