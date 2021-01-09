package experiment

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"

	"github.com/stellentus/cartpoles/lib/agent"
	"github.com/stellentus/cartpoles/lib/config"
	"github.com/stellentus/cartpoles/lib/environment"
	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
	"github.com/stellentus/cartpoles/lib/state"
)

// Execute executes the experiment described by the provided JSON.
func Execute(run uint, conf config.Config, sweepIdx int) error {
	debugLogger := logger.NewDebug(logger.DebugConfig{
		ShouldPrintDebug: true,
	})
	attrs, err := conf.SweptAttributes(sweepIdx)
	agentAttr, envAttr, wrapperAttrs := attrs[0], attrs[1], attrs[2:]
	if err != nil {
		return errors.New("Cannot run sweep: " + err.Error())
	}
	basePath, err := parameterStringify(conf, attrs)
	if err != nil {
		return errors.New("Failed to format path: " + err.Error())
	}
	dataLogger, err := logger.NewData(debugLogger, logger.DataConfig{
		ShouldLogTraces:         conf.Experiment.ShouldLogTraces,
		CacheTracesInRAM:        conf.Experiment.CacheTracesInRAM,
		ShouldLogEpisodeLengths: conf.Experiment.ShouldLogEpisodeLengths,
		BasePath:                basePath,
		FileSuffix:              strconv.Itoa(int(run)),
	})
	if err != nil {
		return errors.New("Could not create data logger: " + err.Error())
	}

	runtime.GOMAXPROCS(conf.MaxCPUs) // Limit the number of CPUs to the provided value (unchanged if the input is <1)

	env, err := InitializeEnvironment(conf.EnvironmentName, run, envAttr, debugLogger)
	if err != nil {
		return errors.New("Could not initialize environment: " + err.Error())
	}

	env, err = InitializeEnvWrapper(conf.WrapperNames, run, wrapperAttrs, env, debugLogger)
	if err != nil {
		err = errors.New("Could not initialize wrapper: " + err.Error())
	}

	agnt, err := InitializeAgent(conf.AgentName, run, agentAttr, env, debugLogger)
	if err != nil {
		return err
	}

	expr, err := New(agnt, env, conf.Experiment, debugLogger, dataLogger)
	if err != nil {
		return err
	}

	_, err = expr.Run()

	return err
}

func InitializeEnvironment(name string, run uint, attr rlglue.Attributes, debug logger.Debug) (rlglue.Environment, error) {
	var err error
	defer debug.Error(&err)

	env, err := environment.Create(name, debug)
	if err != nil {
		err = errors.New("Could not create experiment: " + err.Error())
		return nil, err
	}
	err = env.Initialize(run, attr)
	if err != nil {
		err = errors.New("Could not initialize experiment: " + err.Error())
	}
	return env, err
}

func InitializeAgent(name string, run uint, attr rlglue.Attributes, env rlglue.Environment, debug logger.Debug) (rlglue.Agent, error) {
	var err error
	defer debug.Error(&err)

	agnt, err := agent.Create(name, debug)
	if err != nil {
		err = errors.New("Could not create agent: " + err.Error())
		return nil, err
	}
	err = agnt.Initialize(run, attr, env.GetAttributes())
	if err != nil {
		err = errors.New("Could not initialize agent: " + err.Error())
	}
	return agnt, err
}

func InitializeEnvWrapper(wrapperNames []string, run uint, attr []rlglue.Attributes, env rlglue.Environment, debug logger.Debug) (rlglue.Environment, error) {
	var err error
	defer debug.Error(&err)

	// Return raw environment if there is no parameter to sweep over.
	for i, wrapperName := range wrapperNames {
		env, err = state.Create(wrapperName, env, debug)
		if err != nil {
			err = errors.New("Could not create wrapper: " + err.Error())
			return nil, err
		}
		err = env.Initialize(run, attr[i])
		if err != nil {
			err = errors.New("Could not initialize experiment: " + err.Error())
			return nil, err
		}
	}
	return env, nil
}

func parameterStringify(conf config.Config, attrs []rlglue.Attributes) (string, error) {
	// Generate parameter config.
	sweepAttrMaps := make([]map[string]interface{}, len(attrs))
	for idx, attr := range attrs {
		err := json.Unmarshal(attr, &sweepAttrMaps[idx])
		if err != nil {
			return "", errors.New("Could not parse attributes: " + err.Error())
		}
	}

	paramConfig := struct {
		AgentName       string                   `json:"agent-name"`
		EnvironmentName string                   `json:"environment-name"`
		Agent           map[string]interface{}   `json:"agent-settings"`
		Environment     map[string]interface{}   `json:"environment-settings"`
		StateWrappers   []map[string]interface{} `json:"state-wrappers"`
		Experiment      config.Experiment        `json:"experiment-settings"`
	}{
		conf.AgentName,
		conf.EnvironmentName,
		sweepAttrMaps[0],
		sweepAttrMaps[1],
		sweepAttrMaps[2:],
		conf.Experiment,
	}
	params, err := json.MarshalIndent(&paramConfig, "", "\t")
	if err != nil {
		return "", errors.New("Could not Marshal hyperparameter config: " + err.Error())
	}

	// Get git commit hash.
	gitHashCmd := exec.Command("git", "rev-parse", "--short", "HEAD")
	gitHash, err := gitHashCmd.Output()
	if err != nil {
		return "", errors.New("Could not get the git commit hash: " + err.Error())
	}
	hashPath := strings.Trim(fmt.Sprintf("%s", gitHash), "\n")

	paramHash := sha256.Sum256(params)
	hashPath = fmt.Sprintf("%s_%x", hashPath, paramHash)[:16]
	basePath := fmt.Sprintf("%s/%s", conf.Experiment.DataPath, hashPath)
	if _, err := os.Stat(basePath); os.IsNotExist(err) {
		os.MkdirAll(basePath, os.ModePerm)
	}
	configPath := fmt.Sprintf("%s/config.json", basePath)
	err = ioutil.WriteFile(configPath, params, os.ModePerm)
	if err != nil {
		return "", errors.New("Could not save config json: " + err.Error())
	}
	return basePath, nil
}

func boolToInt(x bool) int {
	if x {
		return 1
	}
	return 0
}

func arrayToString(a []interface{}, delim string) string {
	return strings.Trim(strings.Replace(fmt.Sprint(a), " ", delim, -1), "[]")
}
