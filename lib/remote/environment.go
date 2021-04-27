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

func (env *launcherEnvironment) Initialize(run uint, attr rlglue.Attributes) error {
	port, err := getPort(2200+int(run), attr)
	if err != nil {
		return err
	}

	cc, err := dialGrpc(env.debug, port)
	if err != nil {
		return err
	}
	env.client = NewEnvironmentClient(cc)

	err = launchCommands(run, attr, env.ctx, env.wg)
	if err != nil {
		return err
	}

	err = reattempt(func() error {
		_, err := env.client.Initialize(env.ctx, &EnvironmentAttributes{
			Run:        &Run{Run: uint64(run)},
			Attributes: &Attributes{Attributes: string(attr)},
		})
		return err
	})

	return err
}

// Start returns an initial observation.
func (env *launcherEnvironment) Start(unused bool) (rlglue.State, string) { // TODO fix unused variable
	ctx := context.Background()
	state, _ := env.client.Start(ctx, &Empty{})
	return rlglue.State(state.Values), ""
}

// Step takes an action and provides the new observation, the resulting reward, and whether the state is terminal.
func (env *launcherEnvironment) Step(action rlglue.Action, unused bool) (rlglue.State, float64, bool, string) { // TODO fix unused variable
	ctx := context.Background()
	result, _ := env.client.Step(ctx, &Action{Action: uint64(action.(int))}) // TODO fix the type assertions everywhere
	return rlglue.State(result.State.Values), result.Reward, result.Terminal, ""
}

// GetAttributes returns attributes for this environment.
func (env *launcherEnvironment) GetAttributes() rlglue.Attributes {
	ctx := context.Background()
	attr, _ := env.client.GetAttributes(ctx, &Empty{})
	return rlglue.Attributes(attr.Attributes)
}

func (env *launcherEnvironment) GetInfo(info string, value float64) interface{} {
	return nil
}
