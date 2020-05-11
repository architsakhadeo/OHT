package remote

import (
	"context"
	"encoding/json"
	"errors"
	"os/exec"
	"sync"

	"github.com/stellentus/cartpoles/lib/agent"
	"github.com/stellentus/cartpoles/lib/environment"
	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	"google.golang.org/grpc"
)

func RegisterLaunchers(ctx context.Context, wg *sync.WaitGroup) error {
	// TODO Update this function type to also send rlglue.Attribute for the agent
	err := agent.Add("grpc", func(debug logger.Debug) (rlglue.Agent, error) {
		conn, err := dialGrpc(debug, ":8081")
		if err != nil {
			return nil, err
		}
		return newLauncherAgent(conn, ctx, wg)
	})
	if err != nil {
		return errors.New("failed to initialize grpc agent: " + err.Error())
	}

	err = environment.Add("grpc", func(debug logger.Debug) (rlglue.Environment, error) {
		conn, err := dialGrpc(debug, ":8080")
		if err != nil {
			return nil, err
		}
		return newLauncherEnvironment(conn, ctx, wg)
	})
	if err != nil {
		return errors.New("failed to initialize grpc environment: " + err.Error())
	}
	return nil
}

type launcherAgent struct {
	rlglue.Agent
	ctx context.Context
	wg  *sync.WaitGroup
}

func newLauncherAgent(cc *grpc.ClientConn, ctx context.Context, wg *sync.WaitGroup) (launcherAgent, error) {
	return launcherAgent{
		Agent: NewAgent(cc),
		ctx:   ctx,
		wg:    wg,
	}, nil
}

func (agent launcherAgent) Initialize(experiment, environment rlglue.Attributes) error {
	err := launchCommands(experiment, agent.ctx, agent.wg)
	if err != nil {
		return err
	}
	return agent.Agent.Initialize(experiment, environment)
}

type launcherEnvironment struct {
	rlglue.Environment
	ctx context.Context
	wg  *sync.WaitGroup
}

func newLauncherEnvironment(cc *grpc.ClientConn, ctx context.Context, wg *sync.WaitGroup) (launcherEnvironment, error) {
	return launcherEnvironment{
		Environment: NewEnvironment(cc),
		ctx:         ctx,
		wg:          wg,
	}, nil
}

func (environment launcherEnvironment) Initialize(attr rlglue.Attributes) error {
	err := launchCommands(attr, environment.ctx, environment.wg)
	if err != nil {
		return err
	}
	return environment.Environment.Initialize(attr)
}

// TODO add environment variables by adding a "env" attribute and using `cmd.Env = append(os.Environ(), "PORT=8080", "FOO=actual_value")`
func launchCommands(attr rlglue.Attributes, ctx context.Context, wg *sync.WaitGroup) error {
	extractedAttrs := map[string]json.RawMessage{}
	err := json.Unmarshal(attr, &extractedAttrs)
	if err != nil {
		return errors.New("The gRPC attributes are not valid JSON: " + err.Error())
	}

	commandJson, ok := extractedAttrs["commands"]
	if !ok {
		return nil
	}
	commands := [][]string{}
	err = json.Unmarshal(commandJson, &commands)
	if err != nil {
		return errors.New("The gRPC commands are not valid JSON: " + err.Error())
	}

	for _, command := range commands {
		cmd := exec.CommandContext(ctx, command[0], command[1:]...) // Kills the command based on ctx cancelin
		wg.Add(1)
		err := cmd.Start() // Runs the command in the background
		if err != nil {
			wg.Done()
			return err
		}
		go func() {
			cmd.Wait()
			wg.Done()
		}()
	}

	return nil
}
