package remote

import (
	"context"
	"sync"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

type launcherEnvironment struct {
	client EnvironmentClient
	ctx    context.Context
	wg     *sync.WaitGroup
	debug  logger.Debug
}

func newLauncherEnvironment(debug logger.Debug, ctx context.Context, wg *sync.WaitGroup) (*launcherEnvironment, error) {
	return &launcherEnvironment{
		ctx:   ctx,
		wg:    wg,
		debug: debug,
	}, nil
}

func (env *launcherEnvironment) Initialize(attr rlglue.Attributes) error {
	cc, err := dialGrpc(env.debug, ":8080")
	if err != nil {
		return err
	}
	env.client = NewEnvironmentClient(cc)

	err = launchCommands(attr, env.ctx, env.wg)
	if err != nil {
		return err
	}

	err = reattempt(func() error {
		_, err := env.client.Initialize(env.ctx, &Attributes{Attributes: string(attr)})
		return err
	})

	return err
}

// Start returns an initial observation.
func (env *launcherEnvironment) Start() rlglue.State {
	ctx := context.Background()
	state, _ := env.client.Start(ctx, &Empty{})
	return rlglue.State(state.Values)
}

// Step takes an action and provides the new observation, the resulting reward, and whether the state is terminal.
func (env *launcherEnvironment) Step(action rlglue.Action) (rlglue.State, float64, bool) {
	ctx := context.Background()
	result, _ := env.client.Step(ctx, &Action{Action: uint64(action)})
	return rlglue.State(result.State.Values), result.Reward, result.Terminal
}

// GetAttributes returns attributes for this environment.
func (env *launcherEnvironment) GetAttributes() rlglue.Attributes {
	ctx := context.Background()
	attr, _ := env.client.GetAttributes(ctx, &Empty{})
	return rlglue.Attributes(attr.Attributes)
}
