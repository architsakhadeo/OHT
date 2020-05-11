package remote

import (
	"context"
	"time"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"

	"google.golang.org/grpc"
)

const maxDialAttempts = 200

func dialGrpc(debug logger.Debug, port string) (*grpc.ClientConn, error) {
	var conn *grpc.ClientConn
	err := reattempt(func() error {
		var err error
		conn, err = grpc.Dial("localhost"+port, grpc.WithInsecure())
		return err
	})
	if err != nil {
		debug.Message("err", err)
	}
	return conn, err
}

func reattempt(action func() error) error {
	var err error
	for i := 0; i < maxDialAttempts; i++ {
		err = action()
		if err == nil {
			return nil // It worked!
		}
		time.Sleep(100 * time.Millisecond)
	}
	return err
}

type agentServer struct {
	agent rlglue.Agent
}

func NewAgentServer(agent rlglue.Agent) agentServer {
	return agentServer{agent}
}

func (srv agentServer) Initialize(ctx context.Context, attr *AgentAttributes) (*Empty, error) {
	err := srv.agent.Initialize(rlglue.Attributes(attr.Experiment.Attributes), rlglue.Attributes(attr.Environment.Attributes))
	return &Empty{}, err
}

func (srv agentServer) Start(ctx context.Context, state *State) (*Action, error) {
	action := srv.agent.Start(rlglue.State(state.Values))
	return &Action{Action: uint64(action)}, nil
}

func (srv agentServer) Step(ctx context.Context, result *StepResult) (*Action, error) {
	if result.Terminal {
		srv.agent.End(rlglue.State(result.State.Values), result.Reward)
		return nil, nil
	}

	action := srv.agent.Step(rlglue.State(result.State.Values), result.Reward)
	return &Action{Action: uint64(action)}, nil
}

type agentWrapper struct {
	client AgentClient
}

func NewAgent(cc *grpc.ClientConn) rlglue.Agent {
	return &agentWrapper{NewAgentClient(cc)}
}

func (agent agentWrapper) Initialize(experiment, environment rlglue.Attributes) error {
	ctx := context.Background()
	err := reattempt(func() error {
		_, err := agent.client.Initialize(ctx, &AgentAttributes{
			Experiment:  &Attributes{Attributes: string(experiment)},
			Environment: &Attributes{Attributes: string(environment)},
		})
		return err
	})
	return err
}

func (agent agentWrapper) Start(state rlglue.State) rlglue.Action {
	ctx := context.Background()
	action, _ := agent.client.Start(ctx, &State{Values: []float64(state)})
	return rlglue.Action(action.Action)
}

func (agent agentWrapper) Step(state rlglue.State, reward float64) rlglue.Action {
	ctx := context.Background()
	action, _ := agent.client.Step(ctx, &StepResult{
		State:    &State{Values: []float64(state)},
		Reward:   reward,
		Terminal: false,
	})
	return rlglue.Action(action.Action)
}

func (agent agentWrapper) End(state rlglue.State, reward float64) {
	ctx := context.Background()
	agent.client.Step(ctx, &StepResult{
		State:    &State{Values: []float64(state)},
		Reward:   reward,
		Terminal: true,
	})
}
