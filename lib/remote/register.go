package remote

import (
	"context"
	"encoding/json"
	"errors"
	"os/exec"
	"sync"
	"time"

	"github.com/stellentus/cartpoles/lib/agent"
	"github.com/stellentus/cartpoles/lib/environment"
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

func RegisterLaunchers(ctx context.Context, wg *sync.WaitGroup) error {
	// TODO Update this function type to also send rlglue.Attribute for the agent
	err := agent.Add("grpc", func(debug logger.Debug) (rlglue.Agent, error) {
		return newLauncherAgent(debug, ctx, wg)
	})
	if err != nil {
		return errors.New("failed to initialize grpc agent: " + err.Error())
	}

	err = environment.Add("grpc", func(debug logger.Debug) (rlglue.Environment, error) {
		return newLauncherEnvironment(debug, ctx, wg)
	})
	if err != nil {
		return errors.New("failed to initialize grpc environment: " + err.Error())
	}
	return nil
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
