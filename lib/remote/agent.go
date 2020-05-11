package remote

import (
	"context"
	"sync"

	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

type launcherAgent struct {
	client AgentClient
	ctx    context.Context
	wg     *sync.WaitGroup
	debug  logger.Debug
}

func newLauncherAgent(debug logger.Debug, ctx context.Context, wg *sync.WaitGroup) (*launcherAgent, error) {
	return &launcherAgent{
		ctx:   ctx,
		wg:    wg,
		debug: debug,
	}, nil
}

func (agent *launcherAgent) Initialize(run uint, experiment, environment rlglue.Attributes) error {
	port, err := getPort(2500+int(run), experiment)
	if err != nil {
		return err
	}

	cc, err := dialGrpc(agent.debug, port)
	if err != nil {
		return err
	}
	agent.client = NewAgentClient(cc)

	err = launchCommands(run, experiment, agent.ctx, agent.wg)
	if err != nil {
		return err
	}

	err = reattempt(func() error {
		_, err := agent.client.Initialize(agent.ctx, &AgentAttributes{
			Run:         &Run{Run: uint64(run)},
			Experiment:  &Attributes{Attributes: string(experiment)},
			Environment: &Attributes{Attributes: string(environment)},
		})
		return err
	})
	return err
}

func (agent *launcherAgent) Start(state rlglue.State) rlglue.Action {
	ctx := context.Background()
	action, _ := agent.client.Start(ctx, &State{Values: []float64(state)})
	return rlglue.Action(action.Action)
}

func (agent *launcherAgent) Step(state rlglue.State, reward float64) rlglue.Action {
	ctx := context.Background()
	action, _ := agent.client.Step(ctx, &StepResult{
		State:    &State{Values: []float64(state)},
		Reward:   reward,
		Terminal: false,
	})
	return rlglue.Action(action.Action)
}

func (agent *launcherAgent) End(state rlglue.State, reward float64) {
	ctx := context.Background()
	agent.client.Step(ctx, &StepResult{
		State:    &State{Values: []float64(state)},
		Reward:   reward,
		Terminal: true,
	})
}
