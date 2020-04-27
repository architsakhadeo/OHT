package remote

import (
	"context"

	"github.com/stellentus/cartpoles/go-src/lib/rlglue"
)

type environmentServer struct {
	env rlglue.Environment
}

func NewEnvironmentServer(env rlglue.Environment) EnvironmentServer {
	return environmentServer{env}
}

func (srv environmentServer) Initialize(ctx context.Context, in *Attributes) (*Empty, error) {
	err := srv.env.Initialize(rlglue.Attributes(in.Attributes))
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
