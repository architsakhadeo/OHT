package remote

import (
	"context"

	"github.com/stellentus/cartpoles/lib/rlglue"
)

// These can be used to run gRPC servers. (This is not likely to be used.)

type agentServer struct {
	agent rlglue.Agent
}

func NewAgentServer(agent rlglue.Agent) agentServer {
	return agentServer{agent}
}

func (srv agentServer) Initialize(ctx context.Context, attr *AgentAttributes) (*Empty, error) {
	err := srv.agent.Initialize(uint(attr.Run.Run), rlglue.Attributes(attr.Experiment.Attributes), rlglue.Attributes(attr.Environment.Attributes))
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

type environmentServer struct {
	env rlglue.Environment
}

func NewEnvironmentServer(env rlglue.Environment) EnvironmentServer {
	return environmentServer{env}
}

func (srv environmentServer) Initialize(ctx context.Context, in *EnvironmentAttributes) (*Empty, error) {
	err := srv.env.Initialize(uint(in.Run.Run), rlglue.Attributes(in.Attributes.Attributes))
	return &Empty{}, err
}

func (srv environmentServer) Start(ctx context.Context, in *Empty) (*State, error) {
	state := srv.env.Start()
	return &State{Values: []float64(state)}, nil
}

func (srv environmentServer) Step(ctx context.Context, in *Action) (*StepResult, error) {
	state, reward, terminal := srv.env.Step(rlglue.Action(in.GetAction()))
	return &StepResult{State: &State{Values: []float64(state)}, Reward: reward, Terminal: terminal}, nil
}

func (srv environmentServer) GetAttributes(ctx context.Context, in *Empty) (*Attributes, error) {
	attr := srv.env.GetAttributes()
	return &Attributes{Attributes: string(attr)}, nil
}
