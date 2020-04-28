package remote

import (
	"context"

	"github.com/stellentus/cartpoles/go-src/lib/agent"
	"github.com/stellentus/cartpoles/go-src/lib/logger"
	"github.com/stellentus/cartpoles/go-src/lib/rlglue"

	"google.golang.org/grpc"
)

func init() {
	err := agent.Add("grpc-agent", func(debug logger.Debug) (rlglue.Agent, error) {
		conn, err := grpc.Dial("localhost:8081", grpc.WithInsecure())
		if err != nil {
			debug.Message("err", err)
			return nil, err
		}

		return NewAgent(conn), nil
	})
	if err != nil {
		panic("failed to initialize grpc-agent: " + err.Error())
	}
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
	_, err := agent.client.Initialize(ctx, &AgentAttributes{
		Experiment:  &Attributes{Attributes: string(experiment)},
		Environment: &Attributes{Attributes: string(environment)},
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
