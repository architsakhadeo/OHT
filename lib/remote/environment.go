package remote

import (
	"context"

	"github.com/stellentus/cartpoles/lib/rlglue"

	"google.golang.org/grpc"
)

type environmentWrapper struct {
	client EnvironmentClient
}

func NewEnvironment(cc *grpc.ClientConn) rlglue.Environment {
	return &environmentWrapper{NewEnvironmentClient(cc)}
}

func (env environmentWrapper) Initialize(attr rlglue.Attributes) error {
	ctx := context.Background()
	err := reattempt(func() error {
		_, err := env.client.Initialize(ctx, &Attributes{Attributes: string(attr)})
		return err
	})
	return err
}

// Start returns an initial observation.
func (env environmentWrapper) Start() rlglue.State {
	ctx := context.Background()
	state, _ := env.client.Start(ctx, &Empty{})
	return rlglue.State(state.Values)
}

// Step takes an action and provides the new observation, the resulting reward, and whether the state is terminal.
func (env environmentWrapper) Step(action rlglue.Action) (rlglue.State, float64, bool) {
	ctx := context.Background()
	result, _ := env.client.Step(ctx, &Action{Action: uint64(action)})
	return rlglue.State(result.State.Values), result.Reward, result.Terminal
}

// GetAttributes returns attributes for this environment.
func (env environmentWrapper) GetAttributes() rlglue.Attributes {
	ctx := context.Background()
	attr, _ := env.client.GetAttributes(ctx, &Empty{})
	return rlglue.Attributes(attr.Attributes)
}
