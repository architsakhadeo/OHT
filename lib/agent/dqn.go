package agent

import (
	"encoding/json"
	"math/rand"
	"fmt"
	
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"

	ao "github.com/stellentus/cartpoles/lib/utils/array-opr"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	"github.com/stellentus/cartpoles/lib/utils/buffer"
	nw "github.com/stellentus/cartpoles/lib/utils/network"
)

type Dqn struct {
	logger.Debug
	lastAction          int
	lastState			rlglue.State
	EnableDebug         bool
	NumberOfActions     int `json:"numberOfActions"`
	StateContainsReplay bool `json:"state-contains-replay"`
	gamma				float64 `json:"alpha"`

	state_dim			int `json:"state-len"`
	batch_size			int `json:"batch-size"`

	learning_net		*nw.Vanilla
	target_net			*nw.Vanilla
	loss				tf.Tensor
	optimizer			tf.Tensor
	
	bf					*buffer.Buffer
	bsize				int `json:"buffer-size"`
	btype				string `json:buffer-type`
}

// type Loss func(tf.Tensor, tf.Tensor) float64
// type Optimizer func(float64)

// func mse(out tf.Tensor, target tf.Tensor) tf.Tensor {
// 	res := op.Reduce_sum(op.Square(out - target))
// 	return res
// }

func init() {
	Add("dqn", NewDqn)
}

func NewDqn(logger logger.Debug) (rlglue.Agent, error) {
	return &Dqn{Debug: logger}, nil
}

// Initialize configures the agent with the provided parameters and resets any internal state.
func (agent *Dqn) Initialize(expAttr, envAttr rlglue.Attributes) error {
	var ss struct {
		Seed        int64
		EnableDebug bool `json:"enable-debug"`
	}
	err := json.Unmarshal(expAttr, &ss)
	if err != nil {
		agent.Message("warning", "agent.Example seed wasn't available: "+err.Error())
		ss.Seed = 0
	}
	agent.EnableDebug = ss.EnableDebug

	err = json.Unmarshal(envAttr, &agent)
	if err != nil {
		agent.Message("err", "agent.Example number of Actions wasn't available: "+err.Error())
	}

	rng := rand.New(rand.NewSource(ss.Seed)) // Create a new rand source for reproducibility
	agent.lastAction = rng.Intn(agent.NumberOfActions)

	if agent.EnableDebug {
		agent.Message("msg", "agent.Example Initialize", "seed", ss.Seed, "numberOfActions", agent.NumberOfActions)
	}

	agent.bf = buffer.NewBuffer()
	agent.bf.Initialize(agent.btype, agent.bsize, agent.state_dim)

	// agent.loss = op.ReduceSum(op.Square(out - target))
	agent.optimizer = op.Train.GradientDescentOptimizer(expAttr["alpha"]).minimize(agent.loss)	

	agent.learning_net = nw.NewVanilla()
	agent.learning_net.Initialize(agent.state_dim, agent.NumberOfActions, expAttr["dqn_hidden"], expAttr["dqn_activation"])
	agent.target_net = nw.NewVanilla()
	agent.target_net.Initialize(agent.state_dim, agent.NumberOfActions, expAttr["dqn_hidden"], expAttr["dqn_activation"])

	learning_sess := op.NewScopeWithGraph(agent.learning_net)
	// agent.learning_sess.run(tf.initialize_all_variables())
	agent.learning_net.run(op.Initialize)
	target_sess := op.NewScopeWithGraph(self.target_net)
	agent.SyncTarget()
	
	return nil
}

func (agent *Dqn) Loss(out tf.Tensor, target tf.Tensor) tf.Tensor {
	res := op.ReduceSum(op.Square(out - target))
	return res
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
	agent.Feed(agent.lastState, agent.lastAction, state, reward, agent.gamma)
	agent.Update()
	// agent.lastAction = (agent.lastAction + 1) % agent.NumberOfActions
	agent.lastAction = agent.Policy(state)
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
	samples := agent.bf.Sample(agent.batch_size)
	last_states := samples[:][:agent.state_dim]
	last_actions := samples[:][agent.state_dim]
	states := samples[:][agent.state_dim+1: agent.state_dim*2+1]
	rewards := samples[:][agent.state_dim*2+1]
	gammas := samples[:][agent.state_dim*2+2]

	states = tf.Tensor(states)
	op.StopGradient(states)
	// q_next_all := agent.target_sess.run(agent.target_net, feed_dict={in: states})
	q_next_all, err := agent.target_sess.run(map[tf.Output]*tf.Tensor{in: states}, []tf.Output{out}, nil)
	q_next := ao.RowIndexMax(q_next_all)
	target := ao.ButwiseAdd(rewards, ao.BitwiseMulti(q_next, gammas))

	last_states = tf.Tensor(last_states)
	// q_all := agent.learning_sess.run(agent.learning_net, feed_dict={x: last_states})
	q_all, err := agent.learning_sess.run(agent.learning_net, map[tf.Output]*tf.Tensor{in: last_states, []tf.Output{out}, nil})
	q := ao.RowIndexFloat(q_all, last_actions)
	// agent.learning_sess.run(agent.loss, 
	// 	feed_dict={pred: tf.convert_to_tensor(q, dtype=tf.float32), 
	// 		target: tf.convert_to_tensor(target, dtype=tf.float32)}
	// 	)
	agent.learning_sess.run(agent.optimizer, agent.loss, map[tf.Output]*tf.Tensor{in: last_states, []tf.Output{out}, nil})

}

// Choose action
func (agent *Dqn) Policy(state rlglue.State) int {
	return 0	
}

func (agent *Dqn) SyncTarget() {
	agent.target_net.set_weights(agent.learning_net.get_weights()) 
}

