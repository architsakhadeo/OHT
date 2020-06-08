package experiment

import (
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"github.com/stellentus/cartpoles/lib/agent"
	"github.com/stellentus/cartpoles/lib/config"
	"github.com/stellentus/cartpoles/lib/environment"
	"github.com/stellentus/cartpoles/lib/logger"
	"github.com/stellentus/cartpoles/lib/rlglue"
)

// Execute executes the experiment described by the provided JSON.
func Execute(run uint, conf config.Config, sweepIdx int) error {
	debugLogger := logger.NewDebug(logger.DebugConfig{
		ShouldPrintDebug: true,
	})
	agentAttr, envAttr, err := conf.SweptAttributes(sweepIdx)
	if err != nil {
		return errors.New("Cannot run sweep: " + err.Error())
	}
	savePath, err := hyphenatedStringify(agentAttr, envAttr)
	if err != nil {
		return errors.New("Failed to format path: " + err.Error())
	}
	dataLogger, err := logger.NewData(debugLogger, logger.DataConfig{
		ShouldLogTraces:         conf.Experiment.ShouldLogTraces,
		CacheTracesInRAM:        conf.Experiment.CacheTracesInRAM,
		ShouldLogEpisodeLengths: conf.Experiment.ShouldLogEpisodeLengths,
		BasePath:                fmt.Sprint(conf.Experiment.DataPath, "/", savePath),
		FileSuffix:              strconv.Itoa(int(run)),
	})
	if err != nil {
		return errors.New("Could not create data logger: " + err.Error())
	}

	runtime.GOMAXPROCS(conf.MaxCPUs) // Limit the number of CPUs to the provided value (unchanged if the input is <1)

	environment, err := InitializeEnvironment(conf.EnvironmentName, run, envAttr, debugLogger)
	if err != nil {
		return errors.New("Could not initialize environment: " + err.Error())
	}

	agent, err := InitializeAgent(conf.AgentName, run, agentAttr, environment, debugLogger)
	if err != nil {
		return err
	}

	expr, err := New(agent, environment, conf.Experiment, debugLogger, dataLogger)
	if err != nil {
		return err
	}

	return expr.Run()
}

func InitializeEnvironment(name string, run uint, attr rlglue.Attributes, debug logger.Debug) (rlglue.Environment, error) {
	var err error
	defer debug.Error(&err)

	environment, err := environment.Create(name, debug)
	if err != nil {
		return nil, errors.New("Could not create experiment: " + err.Error())
	}
	err = environment.Initialize(run, attr)
	if err != nil {
		err = errors.New("Could not initialize experiment: " + err.Error())
	}
	environment, err = InitializeEnvWrapper(debug, environment, run, attr)
	if err != nil {
		err = errors.New("Could not initialize wrapper: " + err.Error())
	}
	return environment, err
}

func InitializeAgent(name string, run uint, attr rlglue.Attributes, env rlglue.Environment, debug logger.Debug) (rlglue.Agent, error) {
	var err error
	defer debug.Error(&err)

	agent, err := agent.Create(name, debug)
	if err != nil {
		return nil, errors.New("Could not create agent: " + err.Error())
	}
	err = agent.Initialize(run, attr, env.GetAttributes())
	if err != nil {
		err = errors.New("Could not initialize agent: " + err.Error())
	}
	return agent, err
}

func InitializeEnvWrapper(debug logger.Debug, env rlglue.Environment,
	run uint, attr rlglue.Attributes) (rlglue.Environment, error) {
	// Return raw environment if there is no parameter to sweep over.
	var attrMap map[string]interface{}
	err := json.Unmarshal(attr, &attrMap)
	if err != nil {
		return nil, errors.New("Could not parse attributes: " + err.Error())
	}
	env, err = environment.NewSensorDriftWrapper(debug, env)
	if err != nil {
		return nil, errors.New("Could not create experiment: " + err.Error())
	}
	err = env.Initialize(run, attr)
	if err != nil {
		err = errors.New("Could not initialize experiment: " + err.Error())
	}
	return env, nil
}

func AddSweepAttr(attr rlglue.Attributes, sweepAttr rlglue.Attributes) (rlglue.Attributes, error) {
	// Reformat sweep attributes in environment attributes.
	var attrMap map[string]interface{}
	err := json.Unmarshal(attr, &attrMap)
	if err != nil {
		return nil, errors.New("Could not parse attributes: " + err.Error())
	}
	var sweepAttrMap map[string]interface{}
	err = json.Unmarshal(sweepAttr, &sweepAttrMap)
	if err != nil {
		return nil, errors.New("Could not parse sweep attributes: " + err.Error())
	}
	_, ok := attrMap["sweep"]
	if ok {
		// attrMap["sweep"] = sweepAttrMap
		for k, v := range sweepAttrMap {
			if _, ok = attrMap[k]; !ok {
				attrMap[k] = v
			}
		}
	}
	attr, err = json.Marshal(&attrMap)
	if err != nil {
		return nil, errors.New("Could not encode attributes: " + err.Error())
	}

	return attr, nil
}

func hyphenatedStringify(agentAttr rlglue.Attributes, envAttr rlglue.Attributes) (string, error) {
	pstrings := []string{}
	var sweepAttrMap map[string]interface{}
	err := json.Unmarshal(agentAttr, &sweepAttrMap)
	if err != nil {
		return "", errors.New("Could not parse agent attributes: " + err.Error())
	}
	err = json.Unmarshal(envAttr, &sweepAttrMap)
	if err != nil {
		return "", errors.New("Could not parse environment attributes: " + err.Error())
	}
	for name, value := range sweepAttrMap {
		switch value := value.(type) {
		case int, float64:
			pstrings = append(pstrings, fmt.Sprint(name, "-", value))
		case bool:
			pstrings = append(pstrings, fmt.Sprint(name, "-", boolToInt(value)))
		case []interface{}:
			pstrings = append(pstrings, fmt.Sprint(name, "-", arrayToString(value, ",")))
		default:
			return "", errors.New("Unexpected type")
		}
	}
	// TODO may need a better order.
	sort.Strings(pstrings)
	return strings.Join(pstrings, "_"), nil
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
